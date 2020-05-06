#include <iostream>
#include <fstream>
#include <algorithm>

#include "FortranRoutines.hpp"
#include "ReadWrite.hpp"

#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/vector.h>



int main(int argc, char **argv)
{
	using namespace dealii;

	try
	{
		AssertThrow(argc==2,ExcMessage("Please, pass the directory containing the data!"));

		std::vector<std::string> args;

		for (int i=1; i<argc; ++i)
			args.push_back(argv[i]);

		std::cout << "Starting computation of serial NNLS problem." << std::endl << std::endl;

		dealii::LAPACKFullMatrix<double> matrix_J;

		ReadWrite::read_matrix_HDF5(args[0]+"/matrix_J.h5",matrix_J);

		std::vector<double> tmp_vector_b, tmp_vector_ref_solution;

		ReadWrite::read_vector_HDF5(args[0]+"/vector_b.h5",tmp_vector_b);

		ReadWrite::read_vector_HDF5(args[0]+"/vector_x.h5",tmp_vector_ref_solution);

		dealii::Vector<double> vector_b(tmp_vector_b.size()), ref_solution(tmp_vector_ref_solution.size());

		std::copy(tmp_vector_b.begin(),tmp_vector_b.end(),vector_b.begin());

		std::copy(tmp_vector_ref_solution.begin(),tmp_vector_ref_solution.end(),ref_solution.begin());

		std::cout << "Dimensions of matrix A: " << matrix_J.m() << " x " << matrix_J.n() << std::endl;
		std::cout << "Dimension of vector b: " << vector_b.size() << std::endl;

		dealii::Vector<double> solution(matrix_J.n());

		std::cout << "Start solution ... " << std::flush;
		const double res = Fortran::NNLS(matrix_J,solution,vector_b,1000);
		std::cout << "done" << std::endl;

		dealii::Vector<double> vector_residuum(vector_b);

		matrix_J.vmult(vector_residuum,solution);

		vector_residuum -= vector_b;

		std::cout << "Residuum of NNLS solution: " << vector_residuum.l2_norm() << " / "
				  << vector_b.l2_norm() << " = " << vector_residuum.l2_norm() / vector_b.l2_norm() << std::endl;

		ref_solution -= solution;

		std::cout << "Difference to reference solution: " << ref_solution.l2_norm() << std::endl;

		std::cout << std::endl << std::endl;
	}
	catch (std::exception &exc)
    {
		std::cerr << std::endl
				  << std::endl
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
		std::cerr << std::endl
				  << std::endl
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
