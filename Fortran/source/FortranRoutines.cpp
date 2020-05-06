#include "FortranRoutines.hpp"

extern "C"
{
// two external Fortran routines to solve the non-negative least squares problem

// solves A * X = B  w.r.t  x >= 0 : non-negative least squares
void nnls_(double *A,
		   const int *lda,
		   const int *m,
		   const int *n,
		   double *B,
		   double *X,
		   double *residual_norm,
		   double *work,
		   double *zz,
		   int *index,
		   int *mode,
		   int *nsetp,
		   const int *nmax);

// solves A * X = B  w.r.t  x >= 0 : non-negative least squares
// with trivial modifications to allow for non-positive constraints in array constraints as well
void nnnpls_(double *A,
			 const int *lda,
			 const int *m,
			 const int *n,
			 double *constraints,
			 double *B,
			 double *X,
		     double *residual_norm,
			 double *work,
			 double *zz,
			 int *index,
			 int *mode);
}//extern "C"


namespace Fortran
{

double NNLS(dealii::LAPACKFullMatrix<double> &A,
		  	dealii::Vector<double> &X,
			dealii::Vector<double> &B,
			const unsigned int n_max)
{
	Assert(A.m()==B.size(),dealii::ExcDimensionMismatch(A.m(),B.size()));
	Assert(A.n()==X.size(),dealii::ExcDimensionMismatch(A.n(),X.size()));

	std::vector<double> work_1(A.n()), work_2(A.m());
	std::vector<int> index(A.n());

	const int lda=A.m();
	const int m=A.m();
	const int n=A.n();
	double residual_norm=0;

	/* return value of routine nnls
	   mode == 1: routine computed coefficients successfully
	   mode == 2: dimensions of the problem are wrong
	   mode == 3; iteration count exceeded (more than 3*n iterations)
	*/
	int mode=-1;
	//cardinality of set P
	int nsetp=0;

	const int nmax = (n_max==dealii::numbers::invalid_unsigned_int) ?
					  static_cast<int>(std::min(A.n(),A.m())) :
					  static_cast<int>(n_max);

	nnls_(&A(0,0),&lda,&m,&n,&B[0],&X[0],&residual_norm,work_1.data(),work_2.data(),index.data(),&mode,&nsetp,&nmax);

	if (mode != 1)
	{
		AssertThrow(mode != 2,dealii::ExcMessage("Dimensions of the problem are wrong"));
		AssertThrow(mode != 3,dealii::ExcMessage("Iteration count exceeded (more than 5*n iterations)"));
		AssertThrow(false,dealii::ExcMessage("Unexpected value for mode"));
	}
	return residual_norm;
}

}//namespace Fortran





