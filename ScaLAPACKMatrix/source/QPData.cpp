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


#include <deal.II/base/patterns.h>

#include <deal.II/physics/elasticity/standard_tensors.h>
#include <deal.II/physics/elasticity/kinematics.h>


#include "QPData.hpp"

#include "CompressibleNeoHookean.hpp"
#include "CompressibleOgden.hpp"
#include "CompressibleYeohFleming.hpp"
#include "IncompressibleOgden.hpp"
#include "IncompressibleYeohFleming.hpp"

using namespace dealii;


namespace FE
{

MaterialData::MaterialData(const std::string &material_parameter_file)
{
	ParameterHandler prm;

	declare_parameters(prm);

	prm.parse_input(material_parameter_file);

	parse_parameters(prm);
}


MaterialData::MaterialData(const MaterialData &cpy)
:
material_name(cpy.material_name),
lambda(cpy.lambda),
mu(cpy.mu),
alpha(cpy.alpha),
mu_O(cpy.mu_O),
size(cpy.size),
A(cpy.A),
B(cpy.B),
C(cpy.C),
I_m(cpy.I_m),
alpha_vol(cpy.alpha_vol),
B_vol(cpy.B_vol)
{}


MaterialData&
MaterialData::operator=(const MaterialData &cpy)
{
	if (this != &cpy)
	{
		material_name = cpy.material_name;

		lambda = cpy.lambda;

		mu = cpy.mu;

		alpha = cpy.alpha;

		mu_O = cpy.mu_O;

		size = cpy.size;

		A =cpy.A;

		B = cpy.B;

		C = cpy.C;

		I_m = cpy.I_m;

		alpha_vol = cpy.alpha_vol;

		B_vol = cpy.B_vol;
	}
	return *this;
}


void
MaterialData::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Type of Material");
	{
		Patterns::Selection selection("NeoHookean|Ogden|Yeoh|TNM|Gent|ArrudaBoyce|YeohFleming|IncompressibleOgden|IncompressibleYeohFleming");

	    prm.declare_entry("Name",
	                      "NeoHookean",
	                      selection,
	                      "identifier for material model");
	}
	prm.leave_subsection();

	prm.enter_subsection("Material properties NeoHookean");
	{
		prm.declare_entry("First Lame Parameter",
						  "12",
						  Patterns::Double(0.));

		prm.declare_entry("Second Lame Parameter",
						  "8",
						  Patterns::Double(0.));
	}
	prm.leave_subsection();

	prm.enter_subsection("Material properties Ogden");
	{
		prm.declare_entry("Size",
						  "5",
						  Patterns::Integer(0.));

		prm.declare_entry("Mu 1",
						  "0",
						  Patterns::Double(0.),
						  "in MPa");

		prm.declare_entry("Mu 2",
						  "0",
						  Patterns::Double(0.),
						  "in MPa");

		prm.declare_entry("Mu 3",
						  "0",
						  Patterns::Double(0.),
						  "in MPa");

		prm.declare_entry("Mu 4",
						  "0",
						  Patterns::Double(0.),
						  "in MPa");

		prm.declare_entry("Mu 5",
						  "0",
						  Patterns::Double(0.),
						  "in MPa");

		prm.declare_entry("Alpha 1",
						  "2.875",
						  Patterns::Double(0.));

		prm.declare_entry("Alpha 2",
						  "14.221",
						  Patterns::Double(0.));

		prm.declare_entry("Alpha 3",
						  "1",
						  Patterns::Double(0.));

		prm.declare_entry("Alpha 4",
						  "0",
						  Patterns::Double(0.));

		prm.declare_entry("Alpha 5",
						  "0",
						  Patterns::Double(0.));

		prm.declare_entry("Alpha_vol",
						  "4",
						  Patterns::Double(0.),
						  "in Pa");

		prm.declare_entry("B_vol",
						  "1.3e3",
						  Patterns::Double(0.));
	}
	prm.leave_subsection();

	prm.enter_subsection("Material properties YeohFleming");
	{
		prm.declare_entry("A",
						"0",
						Patterns::Double(0.));

		prm.declare_entry("B",
						"0",
						Patterns::Double(0.));

		prm.declare_entry("C",
						"0",
						Patterns::Double(0.));

		prm.declare_entry("I_m",
						"78.28",
						Patterns::Double(0.));

		prm.declare_entry("Alpha_vol",
						  "4",
						  Patterns::Double(0.),
						  "in Pa");

		prm.declare_entry("B_vol",
						  "1.3e3",
						  Patterns::Double(0.));
	}
	prm.leave_subsection();
}


void
MaterialData::parse_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Type of Material");
	{
		material_name = prm.get("Name");
	}
	prm.leave_subsection();

	if (material_name.compare("NeoHookean") == 0)
	{
		prm.enter_subsection("Material properties NeoHookean");
		{
			lambda = prm.get_double("First Lame Parameter");

	        mu = prm.get_double("Second Lame Parameter");
		}
		prm.leave_subsection();
	}
	else if (material_name.compare("Ogden")==0 || material_name.compare("IncompressibleOgden") == 0)
	{
		prm.enter_subsection("Material properties Ogden");
		{
			size = prm.get_integer("Size");

			Assert(size<=5,ExcMessage("Vector is too long"));

			alpha.reinit(size);

			mu_O.reinit(size);

			B_vol = prm.get_double("B_vol");

			alpha_vol = prm.get_double("Alpha_vol");

			for(int i=0; i<size; ++i)
			{
				alpha[i] = prm.get_double("Alpha "+Utilities::to_string(i+1));

				mu_O[i] = prm.get_double("Mu "+Utilities::to_string(i+1));
			}
		}
		prm.leave_subsection();
	}

	else if (material_name.compare("YeohFleming")==0 || material_name.compare("IncompressibleYeohFleming")== 0)
	{
		prm.enter_subsection("Material properties YeohFleming");
		{
			A = prm.get_double("A");

			B = prm.get_double("B");

			C = prm.get_double("C");

			I_m = prm.get_double("I_m");

			B_vol = prm.get_double("B_vol");

			alpha_vol = prm.get_double("Alpha_vol");
		}
		prm.leave_subsection();
	}
	else if (material_name.compare("Yeoh") == 0)
	{
		Assert(false,ExcMessage("Implementation missing"));
	}
	else if (material_name.compare("TNM") == 0)
	{
		Assert(false,ExcMessage("Implementation missing"));
	}
	else if (material_name.compare("GENT") == 0)
	{
		Assert(false,ExcMessage("Implementation missing"));
	}
	else if (material_name.compare("ArrudaBoyce") == 0)
	{
		Assert(false,ExcMessage("Implementation missing"));
	}
}


template <class Archive>
void
MaterialData::serialize(Archive &ar,
   	   	  	   	   	   	const unsigned int /*version*/)
{
	ar & material_name;

	ar & lambda;

	ar & mu;

	ar & alpha;

	ar & mu_O;

	ar & size;

	ar & A;

	ar & B;

	ar & C;

	ar & I_m;

	ar & alpha_vol;

	ar & B_vol;
}


template <int dim>
QPData<dim>::QPData()
:
tensor_F(Physics::Elasticity::StandardTensors<dim>::I),
det_J(1),
tensor_C(Physics::Elasticity::StandardTensors<dim>::I)
{}


template <int dim>
QPData<dim>::QPData(const MaterialData &material_data)
:
tensor_F(Physics::Elasticity::StandardTensors<dim>::I),
det_J(1),
tensor_C(Physics::Elasticity::StandardTensors<dim>::I)
{
	if (material_data.material_name.compare("NeoHookean")==0)
	{
		material = std::make_shared<ConstitutiveLaws::CompressibleNeoHookean<dim>>(material_data.lambda,
																				   material_data.mu);
	}

	else if (material_data.material_name.compare("Ogden")==0)
	{
		material = std::make_shared<ConstitutiveLaws::CompressibleOgden<dim>>(material_data.alpha,
																			  material_data.mu_O,
																			  material_data.alpha_vol,
																			  material_data.B_vol);
	}

	else if (material_data.material_name.compare("YeohFleming") ==0)
	{
		material = std::make_shared<ConstitutiveLaws::CompressibleYeohFleming<dim>>(material_data.A,
																					material_data.B,
																					material_data.C,
																					material_data.I_m,
																					material_data.alpha_vol,
																					material_data.B_vol);
	}

	else if (material_data.material_name.compare("IncopmressibleYeohFleming") ==0)
	{
		material = std::make_shared<ConstitutiveLaws::IncompressibleYeohFleming<dim>>(material_data.A,
																					  material_data.B,
																					  material_data.C,
																					  material_data.I_m,
																					  material_data.alpha_vol,
																					  material_data.B_vol);
	}

	else if (material_data.material_name.compare("IncompressibleOgden")==0)
	{
		material = std::make_shared<ConstitutiveLaws::IncompressibleOgden<dim>>(material_data.alpha,
																				material_data.mu_O,
																				material_data.alpha_vol,
																				material_data.B_vol);
	}

	else
	{
		AssertThrow(false,ExcMessage("Other material models have to be added"));
	}
}


template <int dim>
void
QPData<dim>::reinit(const MaterialData &material_data)
{
	tensor_F = Physics::Elasticity::StandardTensors<dim>::I;



	tensor_C = Physics::Elasticity::StandardTensors<dim>::I;

	det_J = 1;

	if (material_data.material_name.compare("NeoHookean")==0)
	{
		material = std::make_shared<ConstitutiveLaws::CompressibleNeoHookean<dim>>(material_data.lambda,
																				   material_data.mu);
	}

	else if (material_data.material_name.compare("Ogden")==0)
	{
		material = std::make_shared<ConstitutiveLaws::CompressibleOgden<dim>>(material_data.alpha,
																			  material_data.mu_O,
																			  material_data.alpha_vol,
																			  material_data.B_vol);
	}

	else if (material_data.material_name.compare("YeohFleming") ==0)
	{
		material = std::make_shared<ConstitutiveLaws::CompressibleYeohFleming<dim>>(material_data.A,
																					material_data.B,
																					material_data.C,
																					material_data.I_m,
																					material_data.alpha_vol,
																					material_data.B_vol);
	}

	else if (material_data.material_name.compare("IncopmressibleYeohFleming") ==0)
	{
		material = std::make_shared<ConstitutiveLaws::IncompressibleYeohFleming<dim>>(material_data.A,
																					  material_data.B,
																					  material_data.C,
																					  material_data.I_m,
																					  material_data.alpha_vol,
																					  material_data.B_vol);
	}

	else if (material_data.material_name.compare("IncompressibleOgden")==0)
	{
		material = std::make_shared<ConstitutiveLaws::IncompressibleOgden<dim>>(material_data.alpha,
																				material_data.mu_O,
																				material_data.alpha_vol,
																				material_data.B_vol);
	}

	else
	{
		AssertThrow(false,ExcMessage("Other material models have to be added"));
	}
}


template <int dim>
bool
QPData<dim>::update_values(const Tensor<2,dim> &disp_grad)
{
	tensor_F = Physics::Elasticity::Kinematics::F(disp_grad);

	det_J = determinant(tensor_F);

	if (det_J <= 0)
		return false;

	tensor_C = Physics::Elasticity::Kinematics::C(tensor_F);

	material->stress_S(tensor_S,tensor_C);

	material->material_tangent(tangent,tensor_C);

	return true;
}



template <int dim>
bool
QPData<dim>::update_values(const SymmetricTensor<2,dim> &tensor_C_in)
{
	tensor_C = tensor_C_in;

	det_J = sqrt(determinant(tensor_C));

	if (det_J <= 0)
		return false;

	material->stress_S(tensor_S,tensor_C);

	material->material_tangent(tangent,tensor_C);

	return true;
}


}//namespace FE

#include "QPData.inst"
