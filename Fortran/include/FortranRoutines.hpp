#ifndef LIBS_INCLUDE_FORTRANROUTINES_HPP_
#define LIBS_INCLUDE_FORTRANROUTINES_HPP_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/lapack_full_matrix.h>

namespace Fortran
{

double NNLS(dealii::LAPACKFullMatrix<double> &A,
		  	dealii::Vector<double> &X,
			dealii::Vector<double> &B,
			const unsigned int n_max=dealii::numbers::invalid_unsigned_int);

}//namespace Fortran

#endif /* LIBS_INCLUDE_FORTRANROUTINES_HPP_ */
