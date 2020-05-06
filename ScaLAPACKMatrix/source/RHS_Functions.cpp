/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2019 by the CMFE authors
 *
 * This file is part of the CMFE library.
 *
 * The CMFE library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of CMFE.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Benjamin Brands, Universität Erlangen-Nürnberg, 2019
 */


#include <deal.II/base/exceptions.h>


#include "RHS_Functions.hpp"


using namespace dealii;


namespace FE
{

template <int dim>
Load<dim>::Load(double lf)
:
Function<dim>(dim),
load_factor_(lf)
{}


template <int dim>
ParabolicLoad<dim>::ParabolicLoad(const Point<dim> &origin_,
		  	  	  	  	  	  	  const double p_0_,
								  const double radius_,
								  const double lf)
:
Load<dim>(lf),
origin(origin_),
p_0(p_0_),
radius(radius_)
{}


template <int dim>
double ParabolicLoad<dim>::value(const Point<dim> &p,
								 const unsigned int component) const
{
	Assert(component < dim,ExcIndexRange(component,0,dim));

	if (component == (dim-1))
	{
		const double rel_distance = origin.distance(p) / radius;

		if (rel_distance < 1)
		{
			return - (this->load_factor()) * (p_0 * (1. - rel_distance * rel_distance));
		}
		else
			return 0;
	}
	else
		return 0;
}


template <int dim>
void ParabolicLoad<dim>::value_list(const std::vector<Point<dim>> &point_list,
									std::vector<double> &value_list,
									const unsigned int component) const
{
	 Assert(value_list.size() == point_list.size(), ExcDimensionMismatch(value_list.size(),point_list.size()));
	 Assert(component < dim, ExcIndexRange(component, 0, dim));

	 for (unsigned int i=0; i<point_list.size(); ++i)
		 value_list[i] = ParabolicLoad<dim>::value(point_list[i],component);
}


template <int dim>
void ParabolicLoad<dim>::vector_value(const Point<dim> &p,
									  Vector<double> &value) const
{
	Assert(value.size() == dim, ExcDimensionMismatch(value.size(),dim));

	value = 0;

	value[dim-1] = ParabolicLoad<dim>::value(p,dim-1);
}


template <int dim>
void ParabolicLoad<dim>::vector_value_list(const std::vector<Point<dim>> &point_list,
										   std::vector<Vector<double>> &value_list) const
{
	Assert(value_list.size() == point_list.size(), ExcDimensionMismatch(value_list.size(),point_list.size()));

	for (unsigned int i=0; i<value_list.size(); ++i)
		ParabolicLoad<dim>::vector_value(point_list[i],value_list[i]);
}


template <int dim>
void ParabolicLoad<dim>::tensor_value_list(const std::vector<Point<dim>> &point_list,
										   std::vector<Tensor<1,dim>> &value_list) const
{
	Assert(value_list.size() == point_list.size(), ExcDimensionMismatch(value_list.size(),point_list.size()));

	for (unsigned int i=0; i<value_list.size(); ++i)
		ParabolicLoad<dim>::tensor_value(point_list[i],value_list[i]);
}


template <int dim>
void ParabolicLoad<dim>::tensor_value(const Point<dim> &p,
									  Tensor<1,dim> &t) const
{
	t = 0;

	t[dim-1] = ParabolicLoad<dim>::value(p,dim-1);
}


#include "RHS_Functions.inst"


}//namespace FE
