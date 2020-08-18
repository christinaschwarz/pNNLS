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

		//Test problem
		dealii::LAPACKFullMatrix<double> matrix_J_(5);

		//TEST 1
//		matrix_J_(0,0)=1.;
//		matrix_J_(0,3)=1.;
//		matrix_J_(0,4)=2.;
//		matrix_J_(1,0)=3.;
//		matrix_J_(1,2)=1.;
//		matrix_J_(1,4)=1.;
//		matrix_J_(2,0)=-1.;
//		matrix_J_(2,2)=-1.;
//		matrix_J_(2,3)=-2.;
//		matrix_J_(3,1)=2.;
//		matrix_J_(3,2)=1.;
//		matrix_J_(4,1)=1.;
//		matrix_J_(4,2)=-1.;
//		matrix_J_(4,3)=1.;
//		std::vector<double> tmp_vector_b_={3,5,-2,-1,-2};

		//TEST 2
		matrix_J_(0,0)=1;
		matrix_J_(0,1)=2;
		matrix_J_(0,2)=3;
		matrix_J_(0,3)=0;
		matrix_J_(0,4)=1;

		matrix_J_(1,0)=-1;
		matrix_J_(1,1)=0;
		matrix_J_(1,2)=1;
		matrix_J_(1,3)=-1;
		matrix_J_(1,4)=1;

		matrix_J_(2,0)=1;
		matrix_J_(2,1)=2;
		matrix_J_(2,2)=-1;
		matrix_J_(2,3)=1;
		matrix_J_(2,4)=0;

		matrix_J_(3,0)=2;
		matrix_J_(3,1)=1;
		matrix_J_(3,2)=2;
		matrix_J_(3,3)=3;
		matrix_J_(3,4)=-2;

		matrix_J_(4,0)=-2;
		matrix_J_(4,1)=-1;
		matrix_J_(4,2)=0;
		matrix_J_(4,3)=2;
		matrix_J_(4,4)=-1;
		std::vector<double> tmp_vector_b_={11.,1.,6.,6.,-3.};

		dealii::Vector<double> vector_b_(tmp_vector_b_.size());
		std::copy(tmp_vector_b_.begin(),tmp_vector_b_.end(),vector_b_.begin());
		dealii::Vector<double> solution_(5);
		//dealii::Vector<double> solution_(10);

//		for(unsigned int i=0;i<10;i++){
//			std::cout << vector_b_(i) << std::endl;
//		}
//		for(unsigned int i=0;i<10;i++){
//			for(unsigned int j=0;j<10;j++){
//				std::cout << J_(i,j) << std::endl;
//			}
//		}


		//DO NNLS
		std::cout << "Start solution ... " << std::endl;

		const double res = Fortran::NNLS(matrix_J,solution,vector_b,1000);
		//const double res = Fortran::NNLS(matrix_J_,solution_,vector_b_,100);

		std::cout << "done, x=" << std::endl;
		for(unsigned int i=0;i<5;i++){
			std::cout << solution_(i) << std::endl;
		}
		std::cout << "Rückgabewert NNLS= "<< res << std::endl;

		//calculate Residual
		dealii::Vector<double> vector_residuum(vector_b);
		matrix_J.vmult(vector_residuum,solution);
		vector_residuum -= vector_b;
//		dealii::Vector<double> vector_residuum(vector_b_);
//		matrix_J_.vmult(vector_residuum,solution_);
//		vector_residuum -= vector_b_;

//		std::cout << "Residuum of NNLS solution: " << vector_residuum.l2_norm() << " / "
//				  << vector_b_.l2_norm() << " = " << vector_residuum.l2_norm() / vector_b_.l2_norm() << std::endl;

//		ref_solution -= solution;
//		std::cout << "Difference to reference solution: " << ref_solution.l2_norm() << std::endl;
//		std::cout << std::endl << std::endl;

		int count=0;
		for (unsigned int i=0;i<matrix_J.n();i++){
			if(solution(i)!=0){
				count++;
			}
		}
		std::cout << "anzahl nonnegative elements in x: " << count << std::endl;


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
