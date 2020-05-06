#include <iostream>
#include <fstream>
#include <algorithm>
#include <array>

#include "ReadWrite.hpp"
#include "ScaLAPACKMat.hpp"

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <boost/mpi.hpp>
#include <boost/serialization/array.hpp>



int main (int argc, char **argv)
{
	try
	{
		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv);

		boost::mpi::communicator communicator;

		dealii::deallog.depth_console(0);

		std::vector<std::string> args;
		for (int i=1; i<argc; ++i)
			args.push_back(argv[i]);

		const unsigned int block_size=32;

		dealii::ConditionalOStream pcout(std::cout,(communicator.rank()==0));

		pcout << std::endl << std::endl;

		const std::string data_directory = args[0];

		dealii::LAPACKFullMatrix<double> matrix_J_serial;

		std::array<unsigned int,2> dimensions_J;

		if (communicator.rank()==0)
		{
			ReadWrite::read_matrix_HDF5(args[0]+"/matrix_J.h5",matrix_J_serial);

			dimensions_J[0] = matrix_J_serial.m();

			dimensions_J[1] = matrix_J_serial.n();
		}

		boost::mpi::broadcast(communicator,dimensions_J,0);

		std::shared_ptr<ProcessGrid> grid = std::make_shared<ProcessGrid>(communicator,
																		  dimensions_J[0],
																		  dimensions_J[1],
																		  block_size,
																		  block_size);

		ScaLAPACKMat<double> matrix_J(dimensions_J[0],
									  dimensions_J[1],
									  grid,
									  block_size,
									  block_size);

		pcout << "Dimensions of process grid: "
			  << grid->get_process_grid_rows() << " x " << grid->get_process_grid_columns() << std::endl;

		pcout << "Dimensions of matrix J: "
			  << matrix_J.m() << " x " << matrix_J.n() << std::endl;

		matrix_J.copy_from(matrix_J_serial,0);


		std::vector<double> tmp_vector_b, tmp_vector_ref_solution;

		ReadWrite::read_vector_HDF5(args[0]+"/vector_b.h5",tmp_vector_b);

		ReadWrite::read_vector_HDF5(args[0]+"/vector_x.h5",tmp_vector_ref_solution);

		dealii::Vector<double> vector_b(tmp_vector_b.size()), ref_solution(tmp_vector_ref_solution.size());

		std::copy(tmp_vector_b.begin(),tmp_vector_b.end(),vector_b.begin());

		std::copy(tmp_vector_ref_solution.begin(),tmp_vector_ref_solution.end(),ref_solution.begin());

		pcout << std::endl << std::endl;
    }
	catch (std::exception &exc)
    {
		std::cerr << std::endl << std::endl
				  << "----------------------------------------------------"
                  << std::endl;
		std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
		return 1;
    }
	catch (...)
    {
		std::cerr << std::endl << std::endl
				  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
	return 0;
}
