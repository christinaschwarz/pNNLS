#ifndef dealii_process_grid_h
#define dealii_process_grid_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>



template <typename NumberType>
class ScaLAPACKMat;



    class ProcessGrid
    {
    public:
      template <typename NumberType>
      friend class ScaLAPACKMat;

      ProcessGrid(MPI_Comm           mpi_communicator,
                  const unsigned int n_rows,
                  const unsigned int n_columns);

      ProcessGrid(MPI_Comm           mpi_communicator,
                  const unsigned int n_rows_matrix,
                  const unsigned int n_columns_matrix,
                  const unsigned int row_block_size,
                  const unsigned int column_block_size);
      ProcessGrid(MPI_Comm           mpi_communicator,
                        const std::pair<unsigned int, unsigned int> &grid_dimensions);	//diesen Konstruktor verwenden!!

      ~ProcessGrid();

      unsigned int
      get_process_grid_rows() const;

      unsigned int
      get_process_grid_columns() const;

      int
      get_this_process_row() const;

      int
      get_this_process_column() const;

      template <typename NumberType>
      void
      send_to_inactive(NumberType *value, const int count = 1) const;

      bool
      is_process_active() const;

    private:



      MPI_Comm mpi_communicator;

      MPI_Comm mpi_communicator_inactive_with_root;

      int blacs_context;

      const unsigned int this_mpi_process;

      const unsigned int n_mpi_processes;

      int n_process_rows;

      int n_process_columns;

      int this_process_row;

      int this_process_column;

      bool mpi_process_is_active;
    };



    /*--------------------- Inline functions --------------------------------*/
    inline unsigned int
    ProcessGrid::get_process_grid_rows() const
    {
      return n_process_rows;
    }



    inline unsigned int
    ProcessGrid::get_process_grid_columns() const
    {
      return n_process_columns;
    }



    inline int
    ProcessGrid::get_this_process_row() const
    {
      return this_process_row;
    }



    inline int
    ProcessGrid::get_this_process_column() const
    {
      return this_process_column;
    }



    inline bool
    ProcessGrid::is_process_active() const
    {
      return mpi_process_is_active;
    }



#endif
