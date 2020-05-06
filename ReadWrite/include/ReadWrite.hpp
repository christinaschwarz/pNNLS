#ifndef READWRITE_HPP_
#define READWRITE_HPP_

#include <deal.II/lac/vector.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/lac/lapack_support.h>

#include <boost/filesystem.hpp>
#include <hdf5.h>

#include <string>
#include <vector>



namespace ReadWrite
{
// return hdf5_type_id

template <typename type>
inline hid_t
hdf5_type_id(const type*)
{Assert (false,dealii::ExcNotImplemented()); return -1;}

inline hid_t
hdf5_type_id(const double*)
{return H5T_NATIVE_DOUBLE;}

inline hid_t
hdf5_type_id(const float*)
{return H5T_NATIVE_FLOAT;}

inline hid_t
hdf5_type_id(const int*)
{return H5T_NATIVE_INT;}

inline hid_t
hdf5_type_id(const unsigned int*)
{return H5T_NATIVE_UINT;}

inline hid_t
hdf5_type_id(const char*)
{return H5T_NATIVE_CHAR;}

template<typename type>
void write_matrix_HDF5(const std::string &file_name,
					   const std::vector<type> &matrix,
					   const std::pair<unsigned int,unsigned int> &size,
					   const std::pair<unsigned int,unsigned int> &chunk_sizes=std::make_pair(dealii::numbers::invalid_unsigned_int,
																							  dealii::numbers::invalid_unsigned_int));

template<typename type>
void
write_matrix_HDF5(const std::string &file_name,
				  const dealii::FullMatrix<type> &matrix,
				  const std::pair<unsigned int,unsigned int> &chunk_sizes=std::make_pair(dealii::numbers::invalid_unsigned_int,
																						 dealii::numbers::invalid_unsigned_int));

template<typename type>
void
write_matrix_HDF5(const std::string &file_name,
				  const dealii::LAPACKFullMatrix<type> &matrix,
				  const std::pair<unsigned int,unsigned int> &chunk_sizes=std::make_pair(dealii::numbers::invalid_unsigned_int,
																						 dealii::numbers::invalid_unsigned_int));

template<typename type>
void
write_vector_HDF5(const std::string &file_name,
				  const std::vector<type> &vector,
				  const unsigned int chunk_size=dealii::numbers::invalid_unsigned_int,
				  const bool append_to_file=false);

template<typename type_1,typename type_2>
void
write_matrix_vector_HDF5(const std::string &file_name,
						 const std::vector<type_2> &vector,
						 const std::vector<type_1> &matrix,
						 const std::pair<unsigned int,unsigned int> &size,
						 const std::pair<unsigned int,unsigned int> &chunk_sizes=std::make_pair(dealii::numbers::invalid_unsigned_int,
								 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	 	dealii::numbers::invalid_unsigned_int));

template<typename type>
void
read_matrix_HDF5(const std::string &file_name,
				 std::vector<type> &vector,
				 std::pair<unsigned int,unsigned int> &size);

template<typename type>
void
read_matrix_HDF5(const std::string &file_name,
				 std::vector<type> &vector,
				 const std::pair<unsigned int,unsigned int> &size,
				 const std::pair<unsigned int,unsigned int> &offset);

template<typename type>
void
read_matrix_HDF5(const std::string &file_name,
				 dealii::FullMatrix<type> &matrix,
				 const std::pair<unsigned int,unsigned int> &offset);

template<typename type>
void
read_matrix_HDF5(const std::string &file_name,
				 dealii::LAPACKFullMatrix<type> &matrix,
				 const std::pair<unsigned int,unsigned int> &offset);

template<typename type>
void
read_matrix_HDF5(const std::string &file_name,
				 dealii::FullMatrix<type> &matrix);

template<typename type>
void
read_matrix_HDF5(const std::string &file_name,
				 dealii::LAPACKFullMatrix<type> &matrix);

template<typename type>
void
read_vector_HDF5(const std::string &file_name,
				 std::vector<type> &vector,
				 const unsigned int size=dealii::numbers::invalid_unsigned_int,
				 const unsigned int offset=0);

unsigned int
vector_dimension(const std::string &file_name);

std::pair<unsigned int,unsigned int>
matrix_dimension(const std::string &file_name);

void
append_matrix_state(const std::string &file_name,
					const dealii::LAPACKSupport::State &state);

void
append_matrix_property(const std::string &file_name,
					   const dealii::LAPACKSupport::Property &property);

}//namespace ReadWrite

#endif /* READWRITE_HPP_ */
