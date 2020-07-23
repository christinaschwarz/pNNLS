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

		//pcout << "test2 "  << std::endl;

		//copy b and ref_solution from std::vector to LapackFull matrix
		std::copy(vector_b.begin(),vector_b.end(),b_lapack.begin());
		std::copy(vector_ref_solution.begin(),vector_ref_solution.end(),ref_solution_lapack.begin());

		//process grid erstellen als shared pointer
		std::shared_ptr<ProcessGrid> grid = std::make_shared<ProcessGrid>(communicator, dimensions_J[0], dimensions_J[1], block_size, block_size);

		//pcout << "test3 "  << std::endl;

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
//		pcout << "Dimensions of process grid: " << grid->get_process_grid_rows() << " x " << grid->get_process_grid_columns() << std::endl;
//		pcout << "Dimensions of matrix J: " << J.m() << " x " << J.n() << std::endl;
//		pcout << "Dimensions of vector b: " << J.m() << " x 1" << std::endl;
//		pcout << std::endl << std::endl;


		//TESTS-------------------------------------------------------------------------------------

		//test min und max
//		std::pair <double,std::array<int,2>> minJ=J.min_value(0,dimensions_J[0]-1,0,dimensions_J[1]-1);
//		std::pair <double,std::array<int,2>> maxJ=J.max_value(0,dimensions_J[0]-1,0,dimensions_J[1]-1);
//		std::pair <double,std::array<int,2>> minb=b->min_value(0,dimensions_J[0]-1,0,0);
//		std::pair <double,std::array<int,2>> maxb=b->max_value(0,dimensions_J[0]-1,0,0);
//		pcout << "Jmin_value = " << minJ.first << ", i/j: "<< minJ.second[0] << "/" << minJ.second[1] << std::endl;
//		pcout << "Jmax_value = " << maxJ.first << ", i/j: "<< maxJ.second[0] << "/" << maxJ.second[1] << std::endl;
//		pcout << "bmin_value = " << minb.first << ", i/j: "<< minb.second[0] << "/" << minb.second[1] <<std::endl;
//		pcout << "bmax_value = " << maxb.first << ", i/j: "<< minb.second[0] << "/" << minb.second[1] <<std::endl;



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
//		b_->local_el(0,0)=5.;
//		b_->local_el(1,0)=7.;
//		b_->local_el(2,0)=-8.;
//		b_->local_el(3,0)=7.;
//		b_->local_el(4,0)=1.;

		//TEST 2
		J_.local_el(0,0)=1.;
		J_.local_el(0,1)=2.;
		J_.local_el(0,2)=3.;
		J_.local_el(0,3)=0;
		J_.local_el(0,4)=1.;

		J_.local_el(1,0)=-1.;
		J_.local_el(1,1)=0.;
		J_.local_el(1,2)=1.;
		J_.local_el(1,3)=-1.;
		J_.local_el(1,4)=1.;

		J_.local_el(2,0)=1.;
		J_.local_el(2,1)=2.;
		J_.local_el(2,2)=-1.;
		J_.local_el(2,3)=1.;
		J_.local_el(2,4)=0.;

		J_.local_el(3,0)=2.;
		J_.local_el(3,1)=1.;
		J_.local_el(3,2)=2.;
		J_.local_el(3,3)=3.;
		J_.local_el(3,4)=-2.;

		J_.local_el(4,0)=-2.;
		J_.local_el(4,1)=-1.;
		J_.local_el(4,2)=0.;
		J_.local_el(4,3)=2.;
		J_.local_el(4,4)=-1.;

		b_->local_el(0,0)=11.;
		b_->local_el(1,0)=1.;
		b_->local_el(2,0)=6.;
		b_->local_el(3,0)=6.;
		b_->local_el(4,0)=-3.;


//		for(int i=0;i<5;i++){
//			std::cout << b_->local_el(i,0) << std::endl;
//		}
//		for (int i=0;i<5;i++){
//			for(int j=0;j<5;j++){
//				std::cout << J_.local_el(i,j) << std::endl;
//			}
//		}


		//------------------------------------------------------------------------------------------


		//neue Funktion pNNLS aufrufen
		double tau=0.01;
		int pmax= 100;		//Achtung, pmax sollte kleiner als m=2700 sein
		J.parallel_NNLS(b,x,tau,pmax);
		//J_.parallel_NNLS(b_,x_,tau,pmax);

		//Teilschritte
//		std::vector<int> passive_set={2};
//		//std::cout << "passive set: " <<passive_set.at(0) << std::endl;
//		std::vector<double> tau_(5,0.0);
//		std::shared_ptr<ScaLAPACKMat<double>> Asub  =  std::make_shared<ScaLAPACKMat<double>>(5, 5, grid, 2, 2);
//		J_.copy_to(*Asub);
//		J_.update_qr(Asub, 1, passive_set, tau_);		//this, weil Spalten von A kopiert werden
//		Asub->update_g(b_, b_, 1, 1, tau_);


		//vergleiche x mit ref_solution, Abweichung berechnen
		//.......
		//pcout <<


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
