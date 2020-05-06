/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2010 - 2019 by the CMFE
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


#include <algorithm>
#include <iostream>
#include <fstream>


#include "SimulationParameters.hpp"


using namespace dealii;


namespace FE
{

void FESystemParameters::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Finite element system");
	{
		prm.declare_entry("Polynomial degree",
						  "2",
						  Patterns::Integer(0),
						  "FE polynomial order");
	}
	prm.leave_subsection();
}


void FESystemParameters::parse_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Finite element system");
	{
		poly_degree = prm.get_integer("Polynomial degree");
	}
	prm.leave_subsection();
}


void GeometryParameters::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Geometry");
	{
		prm.declare_entry("Global refinement",
						  "2",
						  Patterns::Integer(0),
						  "Global refinement level");

		prm.declare_entry("Grid scale",
						  "1e-3",
						  Patterns::Double(0.0),
						  "Global grid scaling factor");

		prm.declare_entry("Peak pressure p0",
						  "100",
						  Patterns::Double(0.));

		prm.declare_entry("Load parabola root",
						  "1",
						  Patterns::Double(0.),
						  "Distance between origin and paraboly root");
	}
	prm.leave_subsection();
}

void GeometryParameters::parse_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Geometry");
	{
		global_refinement = prm.get_integer("Global refinement");

		scale = prm.get_double("Grid scale");

		p_0 = prm.get_double("Peak pressure p0");

		parabola_root = prm.get_double("Load parabola root");
	}
	prm.leave_subsection();
}


void LinearSolverParameters::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Linear solver");
	{
		prm.declare_entry("Solver type",
						  "CG",
						  Patterns::Selection("CG|Direct"),
						  "Type of solver used to solve the linear system");

		prm.declare_entry("Residual",
                      	  "1e-6",
						  Patterns::Double(0.0),
						  "Linear solver residual (scaled by residual norm)");

		prm.declare_entry("Max iteration multiplier",
						  "1",
						  Patterns::Double(0.0),
						  "Linear solver iterations (multiples of the system matrix size)");

		prm.declare_entry("Preconditioner type",
                      	  "ssor",
						  Patterns::Selection("jacobi|ssor"),
						  "Type of preconditioner");

		prm.declare_entry("Preconditioner relaxation",
                      	  "0.65",
						  Patterns::Double(0.0),
						  "Preconditioner relaxation value");
	}
	prm.leave_subsection();
}


void LinearSolverParameters::parse_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Linear solver");
	{
		type_lin                  = prm.get("Solver type");

		tol_lin                   = prm.get_double("Residual");

		max_iterations_lin        = prm.get_double("Max iteration multiplier");

		preconditioner_type       = prm.get("Preconditioner type");

		preconditioner_relaxation = prm.get_double("Preconditioner relaxation");
	}
	prm.leave_subsection();
}


void NonlinearSolverParameters::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Nonlinear solver");
	{
		prm.declare_entry("Max iterations Newton-Raphson",
                      	  "10",
						  Patterns::Integer(0),
						  "Number of Newton-Raphson iterations allowed");

		prm.declare_entry("Tolerance force",
                      	  "1.0e-9",
						  Patterns::Double(0.0),
						  "Force residual tolerance");

		prm.declare_entry("Tolerance displacement",
                      	  "1.0e-6",
						  Patterns::Double(0.0),
						  "Displacement error tolerance");
	}
	prm.leave_subsection();
}


void NonlinearSolverParameters::parse_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Nonlinear solver");
	{
		max_iterations_NR = prm.get_integer("Max iterations Newton-Raphson");

		tol_f = prm.get_double("Tolerance force");

		tol_u = prm.get_double("Tolerance displacement");
	}
	prm.leave_subsection();
}


void LoadSteppingParameters::declare_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Load stepping");
	{
		prm.declare_entry("Max number load steps",
                      	  "10",
						  Patterns::Integer(0),
						  "Number of load steps allowed");

		prm.declare_entry("Initial load factor",
                      	  "1",
						  Patterns::Double(0.0));
	}
	prm.leave_subsection();
}


void LoadSteppingParameters::parse_parameters(ParameterHandler &prm)
{
	prm.enter_subsection("Load stepping");
	{
		max_n_load_steps = prm.get_integer("Max number load steps");

		initial_load_factor = prm.get_double("Initial load factor");
	}
	prm.leave_subsection();
}


FEParameters::FEParameters(const std::string &input_file)
{
	ParameterHandler prm;

	declare_parameters(prm);

	prm.parse_input(input_file);

	parse_parameters(prm);
}


void FEParameters::declare_parameters(ParameterHandler &prm)
{
	FESystemParameters::declare_parameters(prm);

	GeometryParameters::declare_parameters(prm);

	LinearSolverParameters::declare_parameters(prm);

	NonlinearSolverParameters::declare_parameters(prm);

	LoadSteppingParameters::declare_parameters(prm);
}


void FEParameters::parse_parameters(ParameterHandler &prm)
{
	FESystemParameters::parse_parameters(prm);

	GeometryParameters::parse_parameters(prm);

	LinearSolverParameters::parse_parameters(prm);

	NonlinearSolverParameters::parse_parameters(prm);

	LoadSteppingParameters::parse_parameters(prm);
}

}
