#ifndef dealii_scalapack_h
#define dealii_scalapack_h

#include <deal.II/base/config.h>

#include <deal.II/base/exceptions.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/thread_management.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/lapack_support.h>

#include <mpi.h>
#include <vector>

#include <memory>

#include "ProcessGrid.hpp"


template <typename NumberType>
class ScaLAPACKMat : protected dealii::TransposeTable<NumberType>
{
public:

  using size_type = unsigned int;

  ScaLAPACKMat(
    const size_type                                           n_rows,
    const size_type                                           n_columns,
    const std::shared_ptr<const ProcessGrid> &process_grid,
    const size_type               row_block_size    = 32,
    const size_type               column_block_size = 32,
    const dealii::LAPACKSupport::Property property = dealii::LAPACKSupport::Property::general);

  ScaLAPACKMat(
    const size_type                                           size,
    const std::shared_ptr<const ProcessGrid> &process_grid,
    const size_type                                           block_size = 32,
    const dealii::LAPACKSupport::Property property = dealii::LAPACKSupport::Property::symmetric);

  ScaLAPACKMat(
    const std::string &                                       filename,
    const std::shared_ptr<const ProcessGrid> &process_grid,
    const size_type row_block_size    = 32,
    const size_type column_block_size = 32);

  ~ScaLAPACKMat() override = default;

  void
  reinit(
    const size_type                                           n_rows,
    const size_type                                           n_columns,
    const std::shared_ptr<const ProcessGrid> &process_grid,
    const size_type               row_block_size    = 32,
    const size_type               column_block_size = 32,
    const dealii::LAPACKSupport::Property property = dealii::LAPACKSupport::Property::general);

  void
  reinit(const size_type                                           size,
         const std::shared_ptr<const ProcessGrid> &process_grid,
         const size_type               block_size = 32,
         const dealii::LAPACKSupport::Property property = dealii::LAPACKSupport::Property::symmetric);

  void
  set_property(const dealii::LAPACKSupport::Property property);

  dealii::LAPACKSupport::Property
  get_property() const;

  dealii::LAPACKSupport::State
  get_state() const;

  ScaLAPACKMat<NumberType> &
  operator=(const dealii::FullMatrix<NumberType> &);

  void
  copy_from(const dealii::LAPACKFullMatrix<NumberType> &matrix,
            const unsigned int                  rank);

  void
  copy_to(dealii::FullMatrix<NumberType> &matrix) const;

  void
  copy_to(dealii::LAPACKFullMatrix<NumberType> &matrix, const unsigned int rank) const;

  void
  copy_to(ScaLAPACKMat<NumberType> &dest) const;

  void
  copy_to(ScaLAPACKMat<NumberType> &                B,
          const std::pair<unsigned int, unsigned int> &offset_A,
          const std::pair<unsigned int, unsigned int> &offset_B,
          const std::pair<unsigned int, unsigned int> &submatrix_size) const;

  void
  copy_transposed(const ScaLAPACKMat<NumberType> &B);

  void
  add(const ScaLAPACKMat<NumberType> &B,
      const NumberType                   a           = 0.,
      const NumberType                   b           = 1.,
      const bool                         transpose_B = false);

  void
  add(const NumberType b, const ScaLAPACKMat<NumberType> &B);

  void
  Tadd(const NumberType b, const ScaLAPACKMat<NumberType> &B);

  void
  mult(const NumberType                   b,
       const ScaLAPACKMat<NumberType> &B,
       const NumberType                   c,
       ScaLAPACKMat<NumberType> &      C,
       const bool                         transpose_A = false,
       const bool                         transpose_B = false) const;

  void
  mmult(ScaLAPACKMat<NumberType> &      C,
        const ScaLAPACKMat<NumberType> &B,
        const bool                         adding = false) const;

  void
  Tmmult(ScaLAPACKMat<NumberType> &      C,
         const ScaLAPACKMat<NumberType> &B,
         const bool                         adding = false) const;

  void
  mTmult(ScaLAPACKMat<NumberType> &      C,
         const ScaLAPACKMat<NumberType> &B,
         const bool                         adding = false) const;

  void
  TmTmult(ScaLAPACKMat<NumberType> &      C,
          const ScaLAPACKMat<NumberType> &B,
          const bool                         adding = false) const;

  void
  save(const std::string &                          filename,
       const std::pair<unsigned int, unsigned int> &chunk_size =
         std::make_pair(dealii::numbers::invalid_unsigned_int,
        		 	 	dealii::numbers::invalid_unsigned_int)) const;

  void
  load(const std::string &filename);

  void
  compute_cholesky_factorization();

  void
  compute_lu_factorization();

  void
  invert();

  std::vector<NumberType>
  eigenpairs_symmetric_by_index(
    const std::pair<unsigned int, unsigned int> &index_limits,
    const bool                                   compute_eigenvectors);

  std::vector<NumberType>
  eigenpairs_symmetric_by_value(
    const std::pair<NumberType, NumberType> &value_limits,
    const bool                               compute_eigenvectors);

  std::vector<NumberType>
  eigenpairs_symmetric_by_index_MRRR(
    const std::pair<unsigned int, unsigned int> &index_limits,
    const bool                                   compute_eigenvectors);

  std::vector<NumberType>
  eigenpairs_symmetric_by_value_MRRR(
    const std::pair<NumberType, NumberType> &value_limits,
    const bool                               compute_eigenvectors);

  std::vector<NumberType>
  compute_SVD(ScaLAPACKMat<NumberType> *U  = nullptr,
              ScaLAPACKMat<NumberType> *VT = nullptr);

  void
  least_squares(ScaLAPACKMat<NumberType> &B, const bool transpose = false);

  unsigned int
  pseudoinverse(const NumberType ratio);

  NumberType
  reciprocal_condition_number(const NumberType a_norm) const;

  NumberType
  l1_norm() const;

  NumberType
  linfty_norm() const;

  NumberType
  frobenius_norm() const;

  size_type
  m() const;

  size_type
  n() const;

  unsigned int
  local_m() const;

  unsigned int
  local_n() const;

  unsigned int
  global_row(const unsigned int loc_row) const;

  unsigned int
  global_column(const unsigned int loc_column) const;

  NumberType
  local_el(const unsigned int loc_row, const unsigned int loc_column) const;

  NumberType &
  local_el(const unsigned int loc_row, const unsigned int loc_column);

  template <class InputVector>
  void
  scale_columns(const InputVector &factors);

  template <class InputVector>
  void
  scale_rows(const InputVector &factors);





  //-----------------------------------------------------------------------------------------------------------------------------------------------------

 void parallel_NNLS
  	(const std::shared_ptr<ScaLAPACKMat<NumberType>> &b, std::shared_ptr<ScaLAPACKMat<NumberType>> &x, const double epsilon, const int pmax, const int max_iterations);

  void update_qr
  	(std::shared_ptr<ScaLAPACKMat<NumberType>> &Asub, const int k, const std::vector<int> passive_set, std::vector<NumberType> &tau);


  void update_g
  	(const std::shared_ptr<ScaLAPACKMat<NumberType>> &b, std::shared_ptr<ScaLAPACKMat<NumberType>> &g, const int k, const int p, std::vector<NumberType> &tau);

  std::pair<NumberType,std::array<int,2>> min_value
  	(const unsigned int row_begin, const unsigned int row_end, const unsigned int col_begin, const unsigned int col_end);

  std::pair<NumberType,std::array<int,2>> max_value
  	(const unsigned int row_begin, const unsigned int row_end, const unsigned int col_begin, const unsigned int col_end);

  void set_element_to_value
  			(const unsigned int row_index, const unsigned int col_index, const NumberType value);

  NumberType return_element
  			(const unsigned int row_index, const unsigned int col_index);


  //-----------------------------------------------------------------------------------------------------------------------------------------------------






private:

  NumberType
  norm_symmetric(const char type) const;

  NumberType
  norm_general(const char type) const;

  std::vector<NumberType>
  eigenpairs_symmetric(
    const bool                                   compute_eigenvectors,
    const std::pair<unsigned int, unsigned int> &index_limits =
      std::make_pair(dealii::numbers::invalid_unsigned_int,
    		  	  	 dealii::numbers::invalid_unsigned_int),
    const std::pair<NumberType, NumberType> &value_limits =
      std::make_pair(std::numeric_limits<NumberType>::quiet_NaN(),
                     std::numeric_limits<NumberType>::quiet_NaN()));

  std::vector<NumberType>
  eigenpairs_symmetric_MRRR(
    const bool                                   compute_eigenvectors,
    const std::pair<unsigned int, unsigned int> &index_limits =
      std::make_pair(dealii::numbers::invalid_unsigned_int,
    		  	  	 dealii::numbers::invalid_unsigned_int),
    const std::pair<NumberType, NumberType> &value_limits =
      std::make_pair(std::numeric_limits<NumberType>::quiet_NaN(),
                     std::numeric_limits<NumberType>::quiet_NaN()));

  void
  save_serial(const std::string &                          filename,
              const std::pair<unsigned int, unsigned int> &chunk_size) const;

  void
  load_serial(const std::string &filename);

  void
  save_parallel(const std::string &                          filename,
                const std::pair<unsigned int, unsigned int> &chunk_size) const;

  void
  load_parallel(const std::string &filename);




  dealii::LAPACKSupport::State state;

  dealii::LAPACKSupport::Property property;

  std::shared_ptr<const ProcessGrid> grid;

  int n_rows;

  int n_columns;

  int row_block_size;

  int column_block_size;

  int n_local_rows;

  int n_local_columns;

  int descriptor[9];

  mutable std::vector<NumberType> work;

  mutable std::vector<int> iwork;

  std::vector<int> ipiv;

  const char uplo;

  const int first_process_row;

  const int first_process_column;

  const int submatrix_row;

  const int submatrix_column;

  mutable dealii::Threads::Mutex mutex;
};

// ----------------------- inline functions ----------------------------

template <typename NumberType>
inline NumberType
ScaLAPACKMat<NumberType>::local_el(const unsigned int loc_row,
                                      const unsigned int loc_column) const
{
  return (*this)(loc_row, loc_column);
}



template <typename NumberType>
inline NumberType &
ScaLAPACKMat<NumberType>::local_el(const unsigned int loc_row,
                                      const unsigned int loc_column)
{
  return (*this)(loc_row, loc_column);
}


template <typename NumberType>
inline unsigned int
ScaLAPACKMat<NumberType>::m() const
{
  return n_rows;
}



template <typename NumberType>
inline unsigned int
ScaLAPACKMat<NumberType>::n() const
{
  return n_columns;
}



template <typename NumberType>
unsigned int
ScaLAPACKMat<NumberType>::local_m() const
{
  return n_local_rows;
}



template <typename NumberType>
unsigned int
ScaLAPACKMat<NumberType>::local_n() const
{
  return n_local_columns;
}




#endif
