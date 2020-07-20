//#include <ScaLAPACKMatrix/include/ProcessGrid.hpp>
#include "ProcessGrid.hpp"
//#include <ScaLAPACKMatrix/include/ScaLAPACK.templates.hpp>
#include "ScaLAPACK.templates.hpp"
//#include <ScaLAPACKMatrix/include/MPI_Tags.hpp>
#include "MPI_Tags.hpp"

#include <deal.II/base/config.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/exceptions.h>



namespace
{

  inline std::pair<int, int>
  compute_processor_grid_sizes(MPI_Comm           mpi_comm,
                               const unsigned int m,
                               const unsigned int n,
                               const unsigned int block_size_m,
                               const unsigned int block_size_n)
  {
    const int n_processes = dealii::Utilities::MPI::n_mpi_processes(mpi_comm);

    const int n_processes_heuristic = int(std::ceil((1. * m) / block_size_m)) *
                                      int(std::ceil((1. * n) / block_size_n));
    const int Np = std::min(n_processes_heuristic, n_processes);

    const double ratio = double(n) / m;
    int          Pc    = static_cast<int>(std::sqrt(ratio * Np));

    int n_process_columns = std::min(Np, std::max(2, Pc));

    int n_process_rows = Np / n_process_columns;

    Assert(n_process_columns >= 1 && n_process_rows >= 1 &&
             n_processes >= n_process_rows * n_process_columns,
		   dealii::ExcMessage(
             "error in process grid: " + std::to_string(n_process_rows) + "x" +
             std::to_string(n_process_columns) + "=" +
             std::to_string(n_process_rows * n_process_columns) + " out of " +
             std::to_string(n_processes)));

    return std::make_pair(n_process_rows, n_process_columns);
  }
} // namespace



    ProcessGrid::ProcessGrid(
      MPI_Comm                                     mpi_comm,
      const std::pair<unsigned int, unsigned int> &grid_dimensions)
      : mpi_communicator(mpi_comm)
      , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_communicator))
      , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_communicator))
      , n_process_rows(grid_dimensions.first)
      , n_process_columns(grid_dimensions.second)
    {
      Assert(grid_dimensions.first > 0,
    		 dealii::ExcMessage("Number of process grid rows has to be positive."));
      Assert(grid_dimensions.second > 0,
    		 dealii::ExcMessage("Number of process grid columns has to be positive."));

      Assert(
        grid_dimensions.first * grid_dimensions.second <= n_mpi_processes,
		dealii::ExcMessage(
          "Size of process grid is larger than number of available MPI processes."));

      const bool column_major = false;

      blacs_context     = Csys2blacs_handle(mpi_communicator);

      const char *order = (column_major ? "Col" : "Row");

      Cblacs_gridinit(&blacs_context, order, n_process_rows, n_process_columns);

      int procrows_ = n_process_rows;
      int proccols_ = n_process_columns;
      Cblacs_gridinfo(blacs_context,
                      &procrows_,
                      &proccols_,
                      &this_process_row,
                      &this_process_column);

      if (this_process_row < 0 || this_process_column < 0)
        mpi_process_is_active = false;
      else
        mpi_process_is_active = true;

      const unsigned int n_active_mpi_processes =
        n_process_rows * n_process_columns;
      Assert(mpi_process_is_active ||
               this_mpi_process >= n_active_mpi_processes,
			 dealii::ExcInternalError());

      std::vector<int> inactive_with_root_ranks;
      inactive_with_root_ranks.push_back(0);
      for (unsigned int i = n_active_mpi_processes; i < n_mpi_processes; ++i)
        inactive_with_root_ranks.push_back(i);

      int       ierr = 0;
      MPI_Group all_group;
      ierr = MPI_Comm_group(mpi_communicator, &all_group);
      AssertThrowMPI(ierr);

      MPI_Group inactive_with_root_group;
      const int n = inactive_with_root_ranks.size();
      ierr        = MPI_Group_incl(all_group,
                            n,
                            inactive_with_root_ranks.data(),
                            &inactive_with_root_group);
      AssertThrowMPI(ierr);

      const int mpi_tag =1345 /*Tags::process_grid_constructor*/;

      ierr = dealii::Utilities::MPI::create_group(mpi_communicator,
                                          	  	  inactive_with_root_group,
												  mpi_tag,
												  &mpi_communicator_inactive_with_root);
      AssertThrowMPI(ierr);

      ierr = MPI_Group_free(&all_group);
      AssertThrowMPI(ierr);
      ierr = MPI_Group_free(&inactive_with_root_group);
      AssertThrowMPI(ierr);


#  ifdef DEBUG
      if (mpi_communicator_inactive_with_root != MPI_COMM_NULL &&
    	  dealii::Utilities::MPI::this_mpi_process(
            mpi_communicator_inactive_with_root) == 0)
        Assert(mpi_process_is_active, dealii::ExcInternalError());
#  endif
    }



    ProcessGrid::ProcessGrid(MPI_Comm           mpi_comm,
                             const unsigned int n_rows_matrix,
                             const unsigned int n_columns_matrix,
                             const unsigned int row_block_size,
                             const unsigned int column_block_size)
      : ProcessGrid(mpi_comm,
                    compute_processor_grid_sizes(mpi_comm,
                                                 n_rows_matrix,
                                                 n_columns_matrix,
                                                 row_block_size,
                                                 column_block_size))
    {}



    ProcessGrid::ProcessGrid(MPI_Comm           mpi_comm,
                             const unsigned int n_rows,
                             const unsigned int n_columns)
      : ProcessGrid(mpi_comm, std::make_pair(n_rows, n_columns))
    {}



    ProcessGrid::~ProcessGrid()
    {
      if (mpi_process_is_active)
        Cblacs_gridexit(blacs_context);

      if (mpi_communicator_inactive_with_root != MPI_COMM_NULL)
        MPI_Comm_free(&mpi_communicator_inactive_with_root);
    }



    template <typename NumberType>
    void
    ProcessGrid::send_to_inactive(NumberType *value, const int count) const
    {
      Assert(count > 0, dealii::ExcInternalError());
      if (mpi_communicator_inactive_with_root != MPI_COMM_NULL)
        {
          const int ierr =
            MPI_Bcast(value,
                      count,
					  dealii::Utilities::MPI::internal::mpi_type_id(value),
                      0 /*from root*/,
                      mpi_communicator_inactive_with_root);
          AssertThrowMPI(ierr);
        }
    }



// instantiations

template void
ProcessGrid::send_to_inactive<double>(double *,
                                      const int) const;
template void
ProcessGrid::send_to_inactive<float>(float *, const int) const;
template void
ProcessGrid::send_to_inactive<int>(int *, const int) const;
