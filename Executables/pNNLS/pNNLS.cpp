#include <iostream>
#include <fstream>
#include <algorithm>
#include <array>
#include <time.h>

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
#include <deal.II/base/timer.h>



int main (int argc, char **argv)
{
	//Time measurement
	dealii::Timer timer;

	try
	{
		//initialize MPI
		dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,argv);
		boost::mpi::communicator communicator;

		dealii::deallog.depth_console(0);	//restrict output on screen to outer loops -->nur ausgabe der obesten ebene, deal2 speziefisch, viele ausgaben vermeiden

		std::vector<std::string> args;
		for (int i=1; i<argc; ++i){		//seichert alle command line arguments im Vektor args ab
			args.push_back(argv[i]);
		}

		const unsigned int block_size=32;		//use a power of 2

		//damit nur der process mit rank 0 die messages ausgibt
		dealii::ConditionalOStream pcout(std::cout,(communicator.rank()==0));
		pcout << std::endl << std::endl;

		const std::string data_directory = args[0];

		//vector b und ref_solution erstellen
		std::vector<double> vector_b;
		std::vector<double> vector_ref_solution;

		//Einlesen der Dateien "vector_b.h5" und "vector_x.h5" in die Vektoren b und ref_solution
		ReadWrite::read_vector_HDF5(args[0]+"/vector_b.h5",vector_b);			//liest std::vector ein
		ReadWrite::read_vector_HDF5(args[0]+"/vector_x.h5",vector_ref_solution);

		//LapackFull-Matrizen für J, b und ref_solution erstellen
		dealii::LAPACKFullMatrix<double> J_lapack;		//J_serial
		std::array<unsigned int,2> dimensions_J;	//leeres array der Größe 2 für dimensionen von J		//was ist ein std::array, vllt unabhängig vom typ???????
		dealii::LAPACKFullMatrix<double> b_lapack;
		dealii::LAPACKFullMatrix<double> ref_solution_lapack;

		communicator.barrier();

		//Einlesen der Datei "matrix_J.h5" in LapackFull-Matrix J
		if (communicator.rank()==0)	{
			ReadWrite::read_matrix_HDF5(args[0]+"/matrix_J.h5",J_lapack); //datei aufrufen
			dimensions_J[0] = J_lapack.m();																	// geht das mit m() und n()????????????
			dimensions_J[1] = J_lapack.n();
//			double min=J_lapack(1,1);
//			double max=J_lapack(1,1);
//			int imax=0;
//			int imin=0;
//			int jmax=0;
//			int jmin=0;
//			std::cout <<"----------------------------" << std::endl;
//			for (unsigned int i=0;i<dimensions_J[0];i++){
//				for (unsigned int j=0; j<dimensions_J[1];j++){
//					if (min>J_lapack(i,j)){
//						min=J_lapack(i,j);
//						imin=i;
//						jmin=j;
//					}
//					if (max<J_lapack(i,j)){
//						max=J_lapack(i,j);
//						imax=i;
//						jmax=j;
//					}
//					//std::cout << J_lapack(i,j)<< std::endl;
//				}
//			}
//			std::cout << "min= " << min << " , i/j: " << imin << " " << jmin << std::endl;
//			std::cout << "max= " << max << " , i/j: " << imax << " " << jmax << std::endl;

//			min=vector_b[0];
//			max=vector_b[0];
//			imax=0;
//			imin=0;
//			std::cout <<"----------------------------" << std::endl;
//			for (unsigned int i=0;i<dimensions_J[0];i++){
//				if (min>vector_b[i]){
//					min=vector_b[i];
//					imin=i;
//				}
//				if (max<vector_b[i]){
//					max=vector_b[i];
//					imax=i;
//				}
//				//std::cout << vector_b[i]<< std::endl;
//			}
//			std::cout << "min= " << min << " , i: " << imin << std::endl;
//			std::cout << "max= " << max << " , i: " << imax << std::endl;
//			std::cout <<"----------------------------" << std::endl;
		}

		communicator.barrier();
		//pcout << "test1 "  << std::endl;
		boost::mpi::broadcast(communicator,dimensions_J,0);	 //allen Prozessen die Dimensionen von J mitteilen
		b_lapack.reinit(dimensions_J[0],1);
		ref_solution_lapack.reinit(dimensions_J[1],1);

		//copy b and ref_solution from std::vector to LapackFull matrix
		std::copy(vector_b.begin(),vector_b.end(),b_lapack.begin());
		std::copy(vector_ref_solution.begin(),vector_ref_solution.end(),ref_solution_lapack.begin());

		//process grid erstellen als shared pointer
		std::shared_ptr<ProcessGrid> grid = std::make_shared<ProcessGrid>(communicator, dimensions_J[0], dimensions_J[1], block_size, block_size);

		// J, b, ref_solution und x als ScalapackMatrix erstellen
		ScaLAPACKMat<double> J(dimensions_J[0], dimensions_J[1], grid, block_size, block_size);	//2tes blocksize=anzahl spalten
		ScaLAPACKMat<double> ref_solution(dimensions_J[1], 1, grid, block_size, 1);
		std::shared_ptr<ScaLAPACKMat<double>> b  =  std::make_shared<ScaLAPACKMat<double>>(dimensions_J[0], 1, grid, block_size, 1);
		std::shared_ptr<ScaLAPACKMat<double>> x  =  std::make_shared<ScaLAPACKMat<double>>(dimensions_J[1], 1, grid, block_size, 1);

		//copy J,b and ref_solution from LapackFullMatrix to ScalapackMatrix  (from locally owned matrix to the distributed matrix)
		J.copy_from(J_lapack,0);		//only process with rank 0 is doing this
		b->copy_from(b_lapack,0);
		ref_solution.copy_from(ref_solution_lapack,0);

		//Bildschirmausgabe der Dimensionen
		pcout << "Dimensions of process grid: " << grid->get_process_grid_rows() << " x " << grid->get_process_grid_columns() << std::endl;
		pcout << "Dimensions of matrix J: " << J.m() << " x " << J.n() << std::endl;
		pcout << "Dimensions of vector b: " << J.m() << " x 1" << std::endl;
		pcout << std::endl << std::endl;


		//TESTS-------------------------------------------------------------------------------------

		std::shared_ptr<ScaLAPACKMat<double>> b_  =  std::make_shared<ScaLAPACKMat<double>>(5, 1, grid, 2, 1);
		ScaLAPACKMat<double> J_(5, 5, grid, 2, 2);
		std::shared_ptr<ScaLAPACKMat<double>> x_  =  std::make_shared<ScaLAPACKMat<double>>(5, 1, grid, 2, 1);

		//TEST 1
//		J_.local_el(0,0)=1.;
//		J_.local_el(0,3)=1.;
//		J_.local_el(0,4)=2.;
//		J_.local_el(1,0)=3.;
//		J_.local_el(1,2)=1.;
//		J_.local_el(1,4)=1.;
//		J_.local_el(2,0)=-1.;
//		J_.local_el(2,2)=-1.;
//		J_.local_el(2,3)=-2.;
//		J_.local_el(3,1)=2.;
//		J_.local_el(3,2)=1.;
//		J_.local_el(4,1)=1.;
//		J_.local_el(4,2)=-1.;
//		J_.local_el(4,3)=1.;
//		b_->local_el(0,0)=3;
//		b_->local_el(1,0)=5;
//		b_->local_el(2,0)=-2.;
//		b_->local_el(3,0)=-1;
//		b_->local_el(4,0)=-2;

		//TEST 2

		J_.set_element_to_value(0,0,1);
		J_.set_element_to_value(0,1,2);
		J_.set_element_to_value(0,2,3);
		J_.set_element_to_value(0,3,0);
		J_.set_element_to_value(0,4,1);

		J_.set_element_to_value(1,0,-1);
		J_.set_element_to_value(1,1,0);
		J_.set_element_to_value(1,2,1);
		J_.set_element_to_value(1,3,-1);
		J_.set_element_to_value(1,4,1);

		J_.set_element_to_value(2,0,1);
		J_.set_element_to_value(2,1,2);
		J_.set_element_to_value(2,2,-1);
		J_.set_element_to_value(2,3,1);
		J_.set_element_to_value(2,4,0);

		J_.set_element_to_value(3,0,2);
		J_.set_element_to_value(3,1,1);
		J_.set_element_to_value(3,2,2);
		J_.set_element_to_value(3,3,3);
		J_.set_element_to_value(3,4,-2);

		J_.set_element_to_value(4,0,-2);
		J_.set_element_to_value(4,1,-1);
		J_.set_element_to_value(4,2,0);
		J_.set_element_to_value(4,3,2);
		J_.set_element_to_value(4,4,-1);

		b_->set_element_to_value(0,0,11);
		b_->set_element_to_value(1,0,1);
		b_->set_element_to_value(2,0,6);
		b_->set_element_to_value(3,0,6);
		b_->set_element_to_value(4,0,-3);

		//test min und max
//		std::pair <double,std::array<int,2>> minJ=J.max_value(0,J.m()-1,0,J.n()-1);
//		std::pair <double,std::array<int,2>> minb=b->max_value(0,J.m()-1,0,0);
//		pcout << "Jmin_value = " << minJ.first << ", i/j: "<< minJ.second[0] << "/" << minJ.second[1] << std::endl;
//		pcout << "bmin_value = " << minb.first << ", i/j: "<< minb.second[0] << "/" << minb.second[1] <<std::endl;
//
//		std::pair <double,std::array<int,2>> minJ3=J_.max_value(0,4,0,4);
//		std::pair <double,std::array<int,2>> minb3=b_->max_value(0,4,0,0);
//		pcout << "Jmin_value = " << minJ3.first << ", i/j: "<< minJ3.second[0] << "/" << minJ3.second[1] << std::endl;
//		pcout << "bmin_value = " << minb3.first << ", i/j: "<< minb3.second[0] << "/" << minb3.second[1] <<std::endl;

		//Test copying whole parts
//		std::pair<unsigned int,unsigned int> offsetA (1,0);
//		std::pair<unsigned int,unsigned int> offsetB (0,0);
//		std::pair<unsigned int,unsigned int> submatrixsize (3,1);
//		J_.copy_to(J_,offsetA,offsetB,submatrixsize);
//		std::cout << "try copying whole parts:" << std::endl;

//		for(int i=0;i<5;i++){
//			for(int j=0;j<5;j++){
//				double a=J_.return_element(i,j);
//				std::cout << a << "/";
//			}
//			std::cout << std::endl;
//		}

		//------------------------------------------------------------------------------------------


		//neue Funktion pNNLS aufrufen
		double tau=1.0e-4;
		int pmax= 1000;		//Achtung, pmax sollte kleiner als m=2700 sein
		J.parallel_NNLS(b,x,tau,pmax,500);
		//J_.parallel_NNLS(b_,x_,tau,pmax,100);


		//calculate Residual
		ScaLAPACKMat<double> residual(dimensions_J[0], 1, grid, block_size, 1);
		J.mmult(residual,*x,false);
		residual.add(*b,1,-1,false);
		pcout << "residual.norm= " << residual.frobenius_norm() << std::endl;
		pcout << "residual.norm/b.norm= " << residual.frobenius_norm()/b->frobenius_norm() << std::endl;

		timer.stop();
		pcout << "Elapsed wall time: " << timer.wall_time() << " seconds.\n";
		std::cout << "Elapsed CPU time on process " << communicator.rank() << " is: " << timer.cpu_time() << " seconds.\n";



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
