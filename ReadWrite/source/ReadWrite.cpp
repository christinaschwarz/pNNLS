#ifndef BOOST_SYSTEM_NO_DEPRECATED
#define BOOST_SYSTEM_NO_DEPRECATED 1
#endif

#include "ReadWrite.hpp"


#include <list>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <deal.II/base/utilities.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/exceptions.h>

#include <boost/algorithm/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>



namespace ReadWrite
{
	using namespace dealii;



	template<typename type>
	void
	write_matrix_HDF5(const std::string &file_name,
					  const std::vector<type> &matrix,
					  const std::pair<unsigned int,unsigned int> &size,
					  const std::pair<unsigned int,unsigned int> &chunk_sizes)
	{
		AssertThrow(size.first*size.second==matrix.size(),
					dealii::ExcMessage("matrix size and given dimensions do not fit"));

		herr_t status;

		hid_t type_id = hdf5_type_id(matrix.data());

		hsize_t dims[2] = {size.first, size.second};
		hsize_t maxdims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
		hsize_t chunk_dims[2];

		if (chunk_sizes.first==dealii::numbers::invalid_unsigned_int || chunk_sizes.second==dealii::numbers::invalid_unsigned_int)
		{
			// get memory consumption of matrix in bytes
			size_t memory_consumption = sizeof(type) * matrix.size();

			// if memory consumption is less than 20 MB, the whole matrix is saved in one chunk
			if (memory_consumption < (1024 * 1024 * 20))
			{
				chunk_dims[0] = size.first;
				chunk_dims[1] = size.second;
			}
			else
			{
				// saving the matrix row-wise
				chunk_dims[0] = 1;
				chunk_dims[1] = size.second;
			}
		}
		else
		{
			chunk_dims[0] = chunk_sizes.first;
			chunk_dims[1] = chunk_sizes.second;
		}
		// create data space with unlimited (extendible) dimensions
		hid_t dataspace = H5Screate_simple(2, dims, maxdims);

		// create a copy of the file access property list
		hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
		// set to use the latest library format
		H5Pset_libver_bounds(fapl_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
		// create a new file using default properties, if the file exists its contents are overwritten
		hid_t file = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file"));

		// enable SWMR writing mode
		//H5Fstart_swmr_write(file);

		// modify dataset creation properties, i.e. enable chunking
		hid_t property = H5Pcreate (H5P_DATASET_CREATE);
		status = H5Pset_chunk (property, 2, chunk_dims);
		AssertThrow(status >= 0, ExcIO());

		// create a new dataset within the file using chunk creation properties
		hid_t dataset = H5Dcreate2 (file, "/matrix", type_id, dataspace, H5P_DEFAULT, property, H5P_DEFAULT);

		// write data to dataset
		status = H5Dwrite (dataset, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix.data());
		AssertThrow(status >= 0, ExcIO());
		// flush data
	/*    status = H5Dflush(dataset);
		AssertThrow(status >= 0, ExcIO());*/

		// close/release resources
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Pclose (property);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (dataspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
		status = H5Pclose(fapl_id);
		AssertThrow(status >= 0, ExcIO());
	}



	template<typename type>
	void
	write_matrix_HDF5(const std::string &file_name,
					  const dealii::FullMatrix<type> &matrix,
					  const std::pair<unsigned int,unsigned int> &chunk_sizes)
	{
		AssertThrow(!matrix.empty(),
					dealii::ExcMessage("matrix must not be empty"));

		herr_t status;
		hid_t type_id = hdf5_type_id(&matrix(0,0));

		hsize_t dims[2] = {matrix.m(), matrix.n()};
		hsize_t maxdims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
		hsize_t chunk_dims[2];

		if (chunk_sizes.first==dealii::numbers::invalid_unsigned_int || chunk_sizes.second==dealii::numbers::invalid_unsigned_int)
		{
			// get memory consumption of matrix in bytes
			size_t memory_consumption = sizeof(type) * matrix.m() * matrix.n();

			// if memory consumption is less than 20 MB, the whole matrix is saved in one chunk
			if (memory_consumption < (1024 * 1024 * 20))
			{
				chunk_dims[0] = matrix.m();
				chunk_dims[1] = matrix.n();
			}
			else
			{
				// saving the matrix row-wise
				chunk_dims[0] = 1;
				chunk_dims[1] = matrix.n();
			}
		}
		else
		{
			chunk_dims[0] = chunk_sizes.first;
			chunk_dims[1] = chunk_sizes.second;
		}
		// create data space with unlimited (extendible) dimensions
		hid_t dataspace = H5Screate_simple(2,dims,maxdims);

		// create a copy of the file access property list
		hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
		// set to use the latest library format
		H5Pset_libver_bounds(fapl_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
		// create a new file using default properties, if the file exists its contents are overwritten
		hid_t file = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file"));

		// enable SWMR writing mode
		//H5Fstart_swmr_write(file);

		// modify dataset creation properties, i.e. enable chunking
		hid_t property = H5Pcreate (H5P_DATASET_CREATE);
		status = H5Pset_chunk (property, 2, chunk_dims);
		AssertThrow(status >= 0, ExcIO());

		// create a new dataset within the file using chunk creation properties
		hid_t dataset = H5Dcreate2 (file, "/matrix", type_id, dataspace, H5P_DEFAULT, property, H5P_DEFAULT);

		// write data to dataset
		status = H5Dwrite (dataset, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &matrix(0,0));
		AssertThrow(status >= 0, ExcIO());
		// flush data
	/*    status = H5Dflush(dataset);
		AssertThrow(status >= 0, ExcIO());*/

		// close/release resources
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Pclose (property);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (dataspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
		status = H5Pclose(fapl_id);
		AssertThrow(status >= 0, ExcIO());
	}



	template<typename type>
	void
	write_matrix_HDF5(const std::string &file_name,
					  const dealii::LAPACKFullMatrix<type> &matrix,
					  const std::pair<unsigned int,unsigned int> &chunk_sizes)
	{
		// LAPACKFullMatrix has Fortran memory layout and therefore we swap variables for rows and columns
		AssertThrow(!matrix.empty(),
					dealii::ExcMessage("matrix must not be empty"));

		herr_t status;
		hid_t type_id = hdf5_type_id(&matrix(0,0));

		hsize_t dims[2] = {matrix.n(), matrix.m()};
		hsize_t maxdims[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
		hsize_t chunk_dims[2];

		if (chunk_sizes.first==dealii::numbers::invalid_unsigned_int || chunk_sizes.second==dealii::numbers::invalid_unsigned_int)
		{
			chunk_dims[0] = matrix.n();
			chunk_dims[1] = matrix.m();

			// get memory consumption of matrix in bytes
			size_t memory_consumption = sizeof(type) * matrix.m() * matrix.n();

			// if memory consumption is less than 20 MB, the whole matrix is saved in one chunk
			if (memory_consumption < (1024 * 1024 * 20))
			{
				chunk_dims[0] = matrix.n();
				chunk_dims[1] = matrix.m();
			}
			else
			{
				// saving the matrix columns (fortran memory layout)
				chunk_dims[0] = 1;
				chunk_dims[1] = matrix.m();
			}
		}
		else
		{
			chunk_dims[0] = chunk_sizes.second;
			chunk_dims[1] = chunk_sizes.first;
		}
		// create data space with unlimited (extendible) dimensions
		hid_t dataspace = H5Screate_simple (2,dims,maxdims);

		// create a copy of the file access property list
		hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
		// set to use the latest library format
		H5Pset_libver_bounds(fapl_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
		// create a new file using default properties, if the file exists its contents are overwritten
		hid_t file = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file"));

		// enable SWMR writing mode
		//H5Fstart_swmr_write(file);

		// modify dataset creation properties, i.e. enable chunking
		hid_t property = H5Pcreate (H5P_DATASET_CREATE);
		status = H5Pset_chunk (property, 2, chunk_dims);
		AssertThrow(status >= 0, ExcIO());

		// create a new dataset within the file using chunk creation properties
		hid_t dataset = H5Dcreate2 (file, "/matrix", type_id, dataspace, H5P_DEFAULT, property, H5P_DEFAULT);

		// write data to dataset
		status = H5Dwrite (dataset, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &matrix(0,0));
		AssertThrow(status >= 0, ExcIO());
		// flush data
	/*    status = H5Dflush(dataset);
		AssertThrow(status >= 0, ExcIO());*/

		// close/release resources
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Pclose (property);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (dataspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
		status = H5Pclose(fapl_id);
		AssertThrow(status >= 0, ExcIO());
	}



	template<typename type_1,typename type_2>
	void
	write_matrix_vector_HDF5(const std::string &file_name,
							 const std::vector<type_2> &vector,
							 const std::vector<type_1> &matrix,
							 const std::pair<unsigned int,unsigned int> &size,
							 const std::pair<unsigned int,unsigned int> &chunk_sizes)
	{
		AssertThrow(size.first*size.second==matrix.size(),
					dealii::ExcMessage("matrix size and given dimensions do not fit"));

		herr_t status;

		hid_t type_id_1 = hdf5_type_id(matrix.data());
		hid_t type_id_2 = hdf5_type_id(vector.data());

		hsize_t dims_1[2] = {size.first, size.second};
		hsize_t maxdims_1[2] = {H5S_UNLIMITED, H5S_UNLIMITED};
		hsize_t dims_2[1] = {vector.size()};
		hsize_t maxdims_2[1] = {H5S_UNLIMITED};
		hsize_t chunk_dims_1[2];
		hsize_t chunk_dims_2[1];

		if (chunk_sizes.first==dealii::numbers::invalid_unsigned_int || chunk_sizes.second==dealii::numbers::invalid_unsigned_int)
		{
			// get memory consumption of matrix in bytes
			size_t memory_consumption = sizeof(type_1) * matrix.size();

			// if memory consumption is less than 20 MB, the whole matrix is saved in one chunk
			if (memory_consumption < (1024 * 1024 * 20))
			{
				chunk_dims_1[0] = size.first;
				chunk_dims_1[1] = size.second;
			}
			else
			{
				// saving the matrix row-wise
				chunk_dims_1[0] = 1;
				chunk_dims_1[1] = size.second;
			}
		}
		else
		{
			chunk_dims_1[0] = chunk_sizes.first;
			chunk_dims_1[1] = chunk_sizes.second;
		}

		if (std::max(chunk_dims_1[0],chunk_dims_1[1]) > vector.size())
			chunk_dims_2[0] = vector.size();
		else
			chunk_dims_2[0] = std::max(chunk_dims_1[0],chunk_dims_1[1]);

		// create data space with unlimited (extendible) dimensions
		hid_t dataspace_1 = H5Screate_simple (2, dims_1, maxdims_1);
		hid_t dataspace_2 = H5Screate_simple (1, dims_2, maxdims_2);

	/*
		// create a copy of the file access property list
		hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);
		// set to use the latest library format
		H5Pset_libver_bounds(fapl_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
		// create a new file using default properties, if the file exists its contents are overwritten
	*/
		hid_t file = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file"+file_name));

		// modify dataset creation properties, i.e. enable chunking
		hid_t property_1 = H5Pcreate(H5P_DATASET_CREATE);
		status = H5Pset_chunk(property_1, 2, chunk_dims_1);
		AssertThrow(status >= 0, ExcIO());

		hid_t property_2 = H5Pcreate(H5P_DATASET_CREATE);
		status = H5Pset_chunk (property_2, 1, chunk_dims_2);
		AssertThrow(status >= 0, ExcIO());

		// create a new dataset within the file using chunk creation properties
		hid_t dataset_1 = H5Dcreate2(file, "/matrix", type_id_1, dataspace_1, H5P_DEFAULT, property_1, H5P_DEFAULT);
		hid_t dataset_2 = H5Dcreate2(file, "/vector", type_id_2, dataspace_2, H5P_DEFAULT, property_2, H5P_DEFAULT);

		// enable SWMR writing mode
		//H5Fstart_swmr_write(file);

		// write data to dataset
		status = H5Dwrite(dataset_1, type_id_1, H5S_ALL, H5S_ALL, H5P_DEFAULT, matrix.data());
		AssertThrow(status >= 0, ExcIO());

		status = H5Dwrite(dataset_2, type_id_2, H5S_ALL, H5S_ALL, H5P_DEFAULT, vector.data());
		AssertThrow(status >= 0, ExcIO());

		// close/release resources
		status = H5Dclose(dataset_1);
		AssertThrow(status >= 0, ExcIO());
		status = H5Dclose(dataset_2);
		AssertThrow(status >= 0, ExcIO());
		status = H5Pclose(property_1);
		AssertThrow(status >= 0, ExcIO());
		status = H5Pclose(property_2);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose(file);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose(dataspace_1);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose(dataspace_2);
		AssertThrow(status >= 0, ExcIO());
	/*    status = H5Pclose(fapl_id);
		AssertThrow(status >= 0, ExcIO());*/
	}



	template<typename type>
	void
	write_into_matrix_HDF5(const std::string &file_name,
						   const std::vector<type> &matrix,
						   const std::pair<unsigned int,unsigned int> &size,
						   const std::pair<unsigned int,unsigned int> &offset)
	{
		herr_t status;
		hid_t type_id = hdf5_type_id(matrix.data());

		// open file
	/*	const std::string hdf5_env_locking = Aux::value_env_variable("HDF5_USE_FILE_LOCKING");
		hid_t file;
		if (boost::iequals(hdf5_env_locking,"FALSE"))
			file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		else
			file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR|H5F_ACC_SWMR_WRITE, H5P_DEFAULT);*/
		hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file"));

		// get data set in file
		hid_t dataset = H5Dopen2(file, "/matrix", H5P_DEFAULT);

		// check the data type of the data in the file
		// data type of source and destination must have the same class
		// see HDF User's Guide: 6.10. Data Transfer: Datatype Conversion and Selection
		hid_t datatype  = H5Dget_type(dataset);
		H5T_class_t t_class_in = H5Tget_class(datatype);
		H5T_class_t t_class = H5Tget_class(type_id);
		AssertThrow(t_class_in == t_class,
					ExcMessage("The data type of the matrix to be read does not match the archive"));

		// determine file space
		hid_t filespace = H5Dget_space (dataset);

		// get number of dimensions in data set
		int rank = H5Sget_simple_extent_ndims (filespace);
		AssertThrow(rank == 2, ExcIO());
		hsize_t dims[2];
		status = H5Sget_simple_extent_dims (filespace, dims, nullptr);
		AssertThrow(status >= 0, ExcIO());

		hsize_t offset_h[2] = {offset.first,offset.second};
		hsize_t size_h[2] = {size.first,size.second};

		AssertThrow(offset.first+size.first<=dims[0],
					dealii::ExcMessage("input parameters size and offset do not fit matrix in file: rows"));
		AssertThrow(offset.second+size.second<=dims[1],
					dealii::ExcMessage("input parameters size and offset do not fit matrix in file: columns"));

		// select hyperslab in the file
		status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_h, nullptr, size_h, nullptr);
		AssertThrow(status >= 0, ExcIO());

		// create a memory data space
		hid_t memspace = H5Screate_simple (rank, size_h, nullptr);

		// write data to file
		status = H5Dwrite (dataset, type_id, memspace, filespace,
						   H5P_DEFAULT, matrix.data());
		AssertThrow(status >= 0, ExcIO());
	/*	status = H5Dflush(dataset);
		AssertThrow(status >= 0, ExcIO());*/

		// close/release resources
		AssertThrow(status >= 0, ExcIO());
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (filespace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (memspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
	}



	template<typename type>
	void
	read_matrix_HDF5(const std::string &file_name,
					 std::vector<type> &matrix,
					 std::pair<unsigned int,unsigned int> &size)
	{
		herr_t status;
		hid_t type_id = hdf5_type_id(matrix.data());

		hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file"+file_name));

		// get data set in file
		hid_t dataset = H5Dopen2(file, "/matrix", H5P_DEFAULT);

		// check the data type of the data in the file
		// data type of source and destination must have the same class
		// see HDF User's Guide: 6.10. Data Transfer: Datatype Conversion and Selection
		hid_t datatype  = H5Dget_type(dataset);
		H5T_class_t t_class_in = H5Tget_class(datatype);
		H5T_class_t t_class = H5Tget_class(type_id);
		AssertThrow(t_class_in == t_class,
					ExcMessage("The data type of the matrix to be read does not match the archive"));

		// determine file space
		hid_t filespace = H5Dget_space (dataset);

		// get number of dimensions in data set
		int rank = H5Sget_simple_extent_ndims (filespace);
		AssertThrow(rank == 2, ExcIO());
		hsize_t dims[2];
		status = H5Sget_simple_extent_dims (filespace, dims, nullptr);
		AssertThrow(status >= 0, ExcIO());

		if (matrix.size() != dims[0]*dims[1])
			matrix.resize(dims[0]*dims[1]);
		size.first = dims[0];
		size.second = dims[1];

		// create a memory data space
		hid_t memspace = H5Screate_simple (rank, dims, nullptr);

		// read data
		status = H5Dread (dataset, type_id, memspace, filespace,
						  H5P_DEFAULT, matrix.data());
		AssertThrow(status >= 0, ExcIO());

		// close/release resources
		AssertThrow(status >= 0, ExcIO());
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (filespace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (memspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
	}



	template<typename type>
	void
	read_matrix_HDF5(const std::string &file_name,
					 std::vector<type> &matrix,
					 const std::pair<unsigned int,unsigned int> &size,
					 const std::pair<unsigned int,unsigned int> &offset)
	{
		herr_t status;
		hid_t type_id = hdf5_type_id(matrix.data());

		// resize matrix array if necessary
		if (matrix.size() != size.first*size.second)
			matrix.resize(size.first*size.second);

		hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file: " + file_name));

		// get data set in file
		hid_t dataset = H5Dopen2(file, "/matrix", H5P_DEFAULT);

		// check the data type of the data in the file
		// data type of source and destination must have the same class
		// see HDF User's Guide: 6.10. Data Transfer: Datatype Conversion and Selection
		hid_t datatype  = H5Dget_type(dataset);
		H5T_class_t t_class_in = H5Tget_class(datatype);
		H5T_class_t t_class = H5Tget_class(type_id);
		AssertThrow(t_class_in == t_class,
					ExcMessage("The data type of the matrix to be read does not match the archive"));

		// determine file space
		hid_t filespace = H5Dget_space (dataset);

		// get number of dimensions in data set
		int rank = H5Sget_simple_extent_ndims (filespace);
		AssertThrow(rank == 2, ExcIO());
		hsize_t dims[2];
		status = H5Sget_simple_extent_dims (filespace, dims, nullptr);
		AssertThrow(status >= 0, ExcIO());

		hsize_t offset_h[2] = {offset.first,offset.second};
		hsize_t size_h[2] = {size.first,size.second};

		AssertThrow(offset.first+size.first<=dims[0],
					dealii::ExcMessage("input parameters size and offset do not fit matrix in file: rows"));
		AssertThrow(offset.second+size.second<=dims[1],
					dealii::ExcMessage("input parameters size and offset do not fit matrix in file: columns"));

		// select hyperslab in the file
		status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_h, nullptr, size_h, nullptr);
		AssertThrow(status >= 0, ExcIO());

		// create a memory data space
		hid_t memspace = H5Screate_simple (rank, size_h, nullptr);

		// read data
		status = H5Dread (dataset, type_id, memspace, filespace,
						  H5P_DEFAULT, matrix.data());
		AssertThrow(status >= 0, ExcIO());

		// close/release resources
		AssertThrow(status >= 0, ExcIO());
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (filespace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (memspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
	}



	template<typename type>
	void
	read_matrix_HDF5(const std::string &file_name,
					 dealii::FullMatrix<type> &matrix,
					 const std::pair<unsigned int,unsigned int> &offset)
	{
		AssertThrow(!matrix.empty(),
					ExcMessage("matrix is empty on input"));

		herr_t status;
		hid_t type_id = hdf5_type_id(&matrix(0,0));

		hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file: " + file_name));

		// get data set in file
		hid_t dataset = H5Dopen2(file, "/matrix", H5P_DEFAULT);

		// check the data type of the data in the file
		// data type of source and destination must have the same class
		// see HDF User's Guide: 6.10. Data Transfer: Datatype Conversion and Selection
		hid_t datatype  = H5Dget_type(dataset);
		H5T_class_t t_class_in = H5Tget_class(datatype);
		H5T_class_t t_class = H5Tget_class(type_id);
		AssertThrow(t_class_in == t_class,
					ExcMessage("The data type of the matrix to be read does not match the archive"));

		// determine file space
		hid_t filespace = H5Dget_space (dataset);

		// get number of dimensions in data set
		int rank = H5Sget_simple_extent_ndims (filespace);
		AssertThrow(rank == 2, ExcIO());
		hsize_t dims[2];
		status = H5Sget_simple_extent_dims (filespace, dims, nullptr);
		AssertThrow(status >= 0, ExcIO());

		hsize_t offset_h[2] = {offset.first,offset.second};
		hsize_t size_h[2] = {matrix.m(),matrix.n()};

		AssertThrow(offset.first+matrix.m()<=dims[0],
					dealii::ExcMessage("input parameters size and offset do not fit matrix in file: rows"));
		AssertThrow(offset.second+matrix.n()<=dims[1],
					dealii::ExcMessage("input parameters size and offset do not fit matrix in file: columns"));

		// select hyperslab in the file
		status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_h, nullptr, size_h, nullptr);
		AssertThrow(status >= 0, ExcIO());

		// create a memory data space
		hid_t memspace = H5Screate_simple (rank, size_h, nullptr);

		// read data
		status = H5Dread (dataset, type_id, memspace, filespace,
						  H5P_DEFAULT, &(matrix(0,0)));
		AssertThrow(status >= 0, ExcIO());

		// close/release resources
		AssertThrow(status >= 0, ExcIO());
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (filespace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (memspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
	}



	template<typename type>
	void
	read_matrix_HDF5(const std::string &file_name,
					 dealii::LAPACKFullMatrix<type> &matrix,
					 const std::pair<unsigned int,unsigned int> &offset)
	{
		AssertThrow(!matrix.empty(),
					ExcMessage("matrix is empty on input"));

		herr_t status;
		hid_t type_id = hdf5_type_id(&matrix(0,0));

		// open file
		hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file: " + file_name));

		// get data set in file
		hid_t dataset = H5Dopen2(file, "/matrix", H5P_DEFAULT);

		// check the data type of the data in the file
		// data type of source and destination must have the same class
		// see HDF User's Guide: 6.10. Data Transfer: Datatype Conversion and Selection
		hid_t datatype  = H5Dget_type(dataset);
		H5T_class_t t_class_in = H5Tget_class(datatype);
		H5T_class_t t_class = H5Tget_class(type_id);
		AssertThrow(t_class_in == t_class,
					ExcMessage("The data type of the matrix to be read does not match the archive"));

		// determine file space
		hid_t filespace = H5Dget_space (dataset);

		// get number of dimensions in data set
		int rank = H5Sget_simple_extent_ndims (filespace);
		AssertThrow(rank == 2, ExcIO());
		hsize_t dims[2];
		status = H5Sget_simple_extent_dims (filespace, dims, nullptr);
		AssertThrow(status >= 0, ExcIO());

		hsize_t offset_h[2] = {offset.first,offset.second};
		//switch rows and columns for Fortran storage format
		hsize_t size_h[2] = {matrix.n(),matrix.m()};

		AssertThrow(offset.first+matrix.n()<=dims[0],
					dealii::ExcMessage("input parameters size and offset do not fit matrix in file: rows"));
		AssertThrow(offset.second+matrix.m()<=dims[1],
					dealii::ExcMessage("input parameters size and offset do not fit matrix in file: columns"));

		// select hyperslab in the file
		status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_h, nullptr, size_h, nullptr);
		AssertThrow(status >= 0, ExcIO());

		// create a memory data space
		hid_t memspace = H5Screate_simple (rank, size_h, nullptr);

		// read data
		status = H5Dread (dataset, type_id, memspace, filespace,
						  H5P_DEFAULT, &(matrix(0,0)));
		AssertThrow(status >= 0, ExcIO());

		// close/release resources
		AssertThrow(status >= 0, ExcIO());
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (filespace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (memspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
	}



	template<typename type>
	void
	read_matrix_HDF5(const std::string &file_name,
					 dealii::FullMatrix<type> &matrix)
	{
		herr_t status;
		type *ptr = nullptr;
		hid_t type_id = hdf5_type_id(ptr);

		// open file
		hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file: " + file_name));

		// get data set in file
		hid_t dataset = H5Dopen2(file, "/matrix", H5P_DEFAULT);

		// check the data type of the data in the file
		// data type of source and destination must have the same class
		// see HDF User's Guide: 6.10. Data Transfer: Datatype Conversion and Selection
		hid_t datatype  = H5Dget_type(dataset);
		H5T_class_t t_class_in = H5Tget_class(datatype);
		H5T_class_t t_class = H5Tget_class(type_id);
		AssertThrow(t_class_in == t_class,
					ExcMessage("The data type of the matrix to be read does not match the archive"));

		// determine file space
		hid_t filespace = H5Dget_space (dataset);

		// get number of dimensions in data set
		int rank = H5Sget_simple_extent_ndims (filespace);
		AssertThrow(rank == 2, ExcIO());
		hsize_t dims[2];
		status = H5Sget_simple_extent_dims (filespace, dims, nullptr);
		AssertThrow(status >= 0, ExcIO());

		if ((matrix.m() != dims[0]) || (matrix.n() != dims[1]))
			matrix.reinit(dims[0],dims[1]);

		// create a memory data space
		hid_t memspace = H5Screate_simple (rank, dims, nullptr);

		// read data
		status = H5Dread (dataset, type_id, memspace, filespace,
						  H5P_DEFAULT, & matrix(0,0));
		AssertThrow(status >= 0, ExcIO());

		// close/release resources
		AssertThrow(status >= 0, ExcIO());
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (filespace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (memspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
	}



	template<typename type>
	void
	read_matrix_HDF5(const std::string &file_name,
					 dealii::LAPACKFullMatrix<type> &matrix)
	{
		herr_t status;
		type *ptr = nullptr;
		hid_t type_id = hdf5_type_id(ptr);

		// open file
		hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file: " + file_name));

		// get data set in file
		hid_t dataset = H5Dopen2(file, "/matrix", H5P_DEFAULT);

		// check the data type of the data in the file
		// data type of source and destination must have the same class
		// see HDF User's Guide: 6.10. Data Transfer: Datatype Conversion and Selection
		hid_t datatype  = H5Dget_type(dataset);
		H5T_class_t t_class_in = H5Tget_class(datatype);
		H5T_class_t t_class = H5Tget_class(type_id);
		AssertThrow(t_class_in == t_class,
					ExcMessage("The data type of the matrix to be read does not match the archive"));

		// determine file space
		hid_t filespace = H5Dget_space (dataset);

		// get number of dimensions in data set
		int rank = H5Sget_simple_extent_ndims (filespace);
		AssertThrow(rank == 2, ExcIO());
		hsize_t dims[2];
		status = H5Sget_simple_extent_dims (filespace, dims, nullptr);
		AssertThrow(status >= 0, ExcIO());

		if (matrix.m() != dims[1] || matrix.n() != dims[0])
			matrix.reinit(dims[1],dims[0]);

		// create a memory data space
		hid_t memspace = H5Screate_simple (rank, dims, nullptr);

		// read data
		status = H5Dread (dataset, type_id, memspace, filespace,
						  H5P_DEFAULT, & matrix(0,0));
		AssertThrow(status >= 0, ExcIO());

		// close/release resources
		AssertThrow(status >= 0, ExcIO());
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (filespace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (memspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
	}



	template<typename type>
	void
	write_vector_HDF5(const std::string &file_name,
					  const std::vector<type> &vector,
					  const unsigned int chunk_size,
					  const bool append_to_file)
	{
		herr_t status;
		hid_t type_id = hdf5_type_id(vector.data());
		hsize_t dims[1] = {vector.size()};
		hsize_t maxdims[1] = {H5S_UNLIMITED};

		// create data space with unlimited (extendible) dimensions
		hid_t dataspace = H5Screate_simple (1, dims, maxdims);

		hid_t file;
		// create a copy of the file access property list
		hid_t fapl_id = H5Pcreate(H5P_FILE_ACCESS);

		if (append_to_file)
		{
			//file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR|H5F_ACC_SWMR_WRITE, H5P_DEFAULT);
			file = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		}
		else
		{
			// set to use the latest library format
			H5Pset_libver_bounds(fapl_id, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST);
			file = H5Fcreate(file_name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
		}
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file"));

		//if append, check whether data set exists
		if (append_to_file)
		{
			// data set exists
			if (H5Lexists(file, "/vector", H5P_DEFAULT) > 0)
				Assert(false,dealii::ExcMessage("data set already exists --> overwriting is forbidden"));
		}
		hsize_t chunk_dims[1];
		if (chunk_size == dealii::numbers::invalid_unsigned_int)
			chunk_dims[0] = vector.size();
		else
			chunk_dims[0] = chunk_size;

		// modify dataset creation properties, i.e. enable chunking
		hid_t property = H5Pcreate (H5P_DATASET_CREATE);
		status = H5Pset_chunk (property, 1, chunk_dims);
		AssertThrow(status >= 0, ExcIO());

		// create a new dataset within the file using chunk creation properties
		hid_t dataset = H5Dcreate2(file, "/vector", type_id, dataspace, H5P_DEFAULT, property, H5P_DEFAULT);

		// write data to dataset
		status = H5Dwrite (dataset, type_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, vector.data());
		AssertThrow(status >= 0, ExcIO());

		// close/release resources
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Pclose (property);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (dataspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
		if (!append_to_file)
		{
			status = H5Pclose(fapl_id);
			AssertThrow(status >= 0, ExcIO());
		}
	}



	template<typename type>
	void
	read_vector_HDF5(const std::string &file_name,
					 std::vector<type> &vector,
					 const unsigned int size,
					 const unsigned int offset)
	{
		herr_t status;
		hid_t type_id = hdf5_type_id(vector.data());

		// open file
		hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file"));

		// get data set in file
		hid_t dataset = H5Dopen2(file, "/vector", H5P_DEFAULT);

		// check the data type of the data in the file
		// data type of source and destination must have the same class
		// see HDF User's Guide: 6.10. Data Transfer: Datatype Conversion and Selection
		hid_t datatype  = H5Dget_type(dataset);
		H5T_class_t t_class_in = H5Tget_class(datatype);
		H5T_class_t t_class = H5Tget_class(type_id);
		AssertThrow(t_class_in == t_class,
					ExcMessage("The data type of the vector to be read does not match the archive"));

		// determine file space
		hid_t filespace = H5Dget_space (dataset);

		// get number of dimensions in data set
		int rank = H5Sget_simple_extent_ndims (filespace);
		AssertThrow(rank == 1, ExcIO());
		hsize_t dims[1];
		status = H5Sget_simple_extent_dims (filespace, dims, nullptr);
		AssertThrow(status >= 0, ExcIO());

		hsize_t offset_h[1] = {offset};
		hsize_t size_h[1];

		// read whole vector stored in file
		if (size==dealii::numbers::invalid_unsigned_int)
		{
			size_h[0] = dims[0];

			if (vector.size() != dims[0])
				vector.resize(dims[0]);
		}
		//read only part of the vector
		else
		{
			size_h[0] = size;

			if (vector.size() != size)
				vector.resize(size);

			AssertThrow(offset+size<=dims[0],
						dealii::ExcMessage("input parameters size and offset do not fit vector in file"));
		}

		// select hyperslab in the file only if part of the vector has to be read
		if (size!=dealii::numbers::invalid_unsigned_int)
		{
			status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset_h, nullptr, size_h, nullptr);
			AssertThrow(status >= 0, ExcIO());
		}

		// create a memory data space
		hid_t memspace = H5Screate_simple (rank, size_h, nullptr);

		// read data
		status = H5Dread(dataset, type_id, memspace, filespace,
						 H5P_DEFAULT, vector.data());
		AssertThrow(status >= 0, ExcIO());

		// close/release resources
		status = H5Sclose (memspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (filespace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());
	}



	unsigned int
	vector_dimension(const std::string &file_name)
	{
		herr_t status;

		// open file in read mode
		//hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY|H5F_ACC_SWMR_READ, H5P_DEFAULT);
		hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file"));

		// get data set in file
		hid_t dataset = H5Dopen2(file, "/vector", H5P_DEFAULT);
		// data set "/vector" does not exist --> return 0 as dimension
		if (dataset < 0)
		{
			status = H5Fclose (file);
			AssertThrow(status >= 0, ExcIO());
			return 0;
		}
		// determine file space
		hid_t filespace = H5Dget_space(dataset);

		// get number of dimensions in data set
		int rank = H5Sget_simple_extent_ndims(filespace);
		AssertThrow(rank == 1, ExcIO());
		hsize_t dims[1];
		status = H5Sget_simple_extent_dims(filespace, dims, nullptr);
		AssertThrow(status >= 0, ExcIO());

		// close/release resources
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (filespace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());

		return dims[0];
	}



	std::pair<unsigned int,unsigned int>
	matrix_dimension(const std::string &file_name)
	{
		herr_t status;

		// open file in read mode
		//hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY|H5F_ACC_SWMR_READ, H5P_DEFAULT);
		hid_t file = H5Fopen(file_name.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
		AssertThrow(file>=0,dealii::ExcMessage("Cannot open file"));

		// file does not exist --> return 0 as dimension
		if (file < 0)
			return std::make_pair((unsigned int)0,(unsigned int)0);

		// get data set in file
		hid_t dataset = H5Dopen2(file, "/matrix", H5P_DEFAULT);
		// data set "/matrix" does not exist --> return 0 as dimension
		if (dataset < 0)
		{
			status = H5Fclose (file);
			AssertThrow(status >= 0, ExcIO());
			return std::make_pair((unsigned int)0,(unsigned int)0);
		}
		// determine file space
		hid_t filespace = H5Dget_space(dataset);

		// get number of dimensions in data set
		int rank = H5Sget_simple_extent_ndims(filespace);
		AssertThrow(rank == 2, ExcIO());
		hsize_t dims[2];
		status = H5Sget_simple_extent_dims(filespace, dims, nullptr);
		AssertThrow(status >= 0, ExcIO());

		// close/release resources
		status = H5Dclose (dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose (filespace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose (file);
		AssertThrow(status >= 0, ExcIO());

		return std::make_pair((unsigned int)dims[0],(unsigned int)dims[1]);
	}



	void create_HDF5_state_enum(hid_t &state_enum_id)
	{
		// create HDF5 enum type for dealii::LAPACKSupport::State
		dealii::LAPACKSupport::State val;
		state_enum_id = H5Tcreate (H5T_ENUM, sizeof(dealii::LAPACKSupport::State));
		val = dealii::LAPACKSupport::State::cholesky;
		herr_t status = H5Tenum_insert (state_enum_id, "cholesky", (int *)&val);
		AssertThrow(status >= 0, ExcInternalError());
		val = dealii::LAPACKSupport::State::eigenvalues;
		status = H5Tenum_insert (state_enum_id, "eigenvalues", (int *)&val);
		AssertThrow(status >= 0, ExcInternalError());
		val = dealii::LAPACKSupport::State::inverse_matrix;
		status = H5Tenum_insert (state_enum_id, "inverse_matrix", (int *)&val);
		AssertThrow(status >= 0, ExcInternalError());
		val = dealii::LAPACKSupport::State::inverse_svd;
		status = H5Tenum_insert (state_enum_id, "inverse_svd", (int *)&val);
		AssertThrow(status >= 0, ExcInternalError());
		val = dealii::LAPACKSupport::State::lu;
		status = H5Tenum_insert (state_enum_id, "lu", (int *)&val);
		AssertThrow(status >= 0, ExcInternalError());
		val = dealii::LAPACKSupport::State::matrix;
		status = H5Tenum_insert (state_enum_id, "matrix", (int *)&val);
		AssertThrow(status >= 0, ExcInternalError());
		val = dealii::LAPACKSupport::State::svd;
		status = H5Tenum_insert (state_enum_id, "svd", (int *)&val);
		AssertThrow(status >= 0, ExcInternalError());
		val = dealii::LAPACKSupport::State::unusable;
		status = H5Tenum_insert (state_enum_id, "unusable", (int *)&val);
		AssertThrow(status >= 0, ExcInternalError());
	}



	void create_HDF5_property_enum(hid_t &property_enum_id)
	{
		// create HDF5 enum type for LAPACKSupport::Property
		property_enum_id = H5Tcreate (H5T_ENUM, sizeof(dealii::LAPACKSupport::Property));
		dealii::LAPACKSupport::Property prop = dealii::LAPACKSupport::Property::diagonal;
		herr_t status = H5Tenum_insert (property_enum_id, "diagonal", (int *)&prop);
		AssertThrow(status >= 0, ExcInternalError());
		prop = dealii::LAPACKSupport::Property::general;
		status = H5Tenum_insert (property_enum_id, "general", (int *)&prop);
		AssertThrow(status >= 0, ExcInternalError());
		prop = dealii::LAPACKSupport::Property::hessenberg;
		status = H5Tenum_insert (property_enum_id, "hessenberg", (int *)&prop);
		AssertThrow(status >= 0, ExcInternalError());
		prop = dealii::LAPACKSupport::Property::lower_triangular;
		status = H5Tenum_insert (property_enum_id, "lower_triangular", (int *)&prop);
		AssertThrow(status >= 0, ExcInternalError());
		prop = dealii::LAPACKSupport::Property::symmetric;
		status = H5Tenum_insert (property_enum_id, "symmetric", (int *)&prop);
		AssertThrow(status >= 0, ExcInternalError());
		prop = dealii::LAPACKSupport::Property::upper_triangular;
		status = H5Tenum_insert (property_enum_id, "upper_triangular", (int *)&prop);
		AssertThrow(status >= 0, ExcInternalError());
	}



	void
	append_matrix_state(const std::string &file_name,
						const dealii::LAPACKSupport::State &state)
	{
		// open file in read/write mode
		//hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDWR|H5F_ACC_SWMR_WRITE, H5P_DEFAULT);
		hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		AssertThrow(file_id>=0,dealii::ExcMessage("Cannot open file"));

		// create HDF5 enum type for LAPACKSupport::State and LAPACKSupport::Property
		hid_t state_enum_id;
		create_HDF5_state_enum(state_enum_id);

		// create the data space for the state enum
		hsize_t dims_state[1];
		dims_state[0]=1;
		hid_t state_enum_dataspace = H5Screate_simple(1, dims_state, nullptr);
		// create the dataset for the state enum
		hid_t state_enum_dataset = H5Dcreate2(file_id, "/state", state_enum_id, state_enum_dataspace,
											  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		// write the dataset for the state enum
		herr_t status = H5Dwrite(state_enum_dataset, state_enum_id,
								 H5S_ALL, H5S_ALL, H5P_DEFAULT,
								 &state);
		AssertThrow(status >= 0, ExcIO());
	/*    status = H5Dflush(state_enum_dataset);
		AssertThrow(status >= 0, ExcIO());*/

		// close/release resources
		status = H5Dclose(state_enum_dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose(state_enum_dataspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Tclose(state_enum_id);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose(file_id);
		AssertThrow(status >= 0, ExcIO());
	}



	void append_matrix_property(const std::string &file_name,
								const dealii::LAPACKSupport::Property &property)
	{
		// open file in read/write mode
		//hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDWR|H5F_ACC_SWMR_WRITE, H5P_DEFAULT);
		hid_t file_id = H5Fopen(file_name.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
		AssertThrow(file_id>=0,dealii::ExcMessage("Cannot open file"));

		// create HDF5 enum type for LAPACKSupport::State and LAPACKSupport::Property
		hid_t property_enum_id;
		create_HDF5_property_enum(property_enum_id);

		// create the data space for the property enum
		hsize_t dims_property[1];
		dims_property[0]=1;
		hid_t property_enum_dataspace = H5Screate_simple(1, dims_property, nullptr);
		// create the dataset for the property enum
		hid_t property_enum_dataset = H5Dcreate2(file_id, "/property", property_enum_id, property_enum_dataspace,
												 H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		// write the dataset for the property enum
		herr_t status = H5Dwrite(property_enum_dataset, property_enum_id,
								 H5S_ALL, H5S_ALL, H5P_DEFAULT,
								 &property);
		AssertThrow(status >= 0, ExcIO());
	/*    status = H5Dflush(property_enum_dataset);
		AssertThrow(status >= 0, ExcIO());*/

		// close/release resources
		status = H5Dclose(property_enum_dataset);
		AssertThrow(status >= 0, ExcIO());
		status = H5Sclose(property_enum_dataspace);
		AssertThrow(status >= 0, ExcIO());
		status = H5Tclose(property_enum_id);
		AssertThrow(status >= 0, ExcIO());
		status = H5Fclose(file_id);
		AssertThrow(status >= 0, ExcIO());
	}

}//namespace ReadWrite

#include "ReadWrite.inst"
