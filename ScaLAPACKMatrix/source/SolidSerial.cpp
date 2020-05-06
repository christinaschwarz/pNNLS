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

#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/quadrature.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_face.h>

#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_selector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition_selector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/physics/elasticity/kinematics.h>
#include <deal.II/physics/elasticity/standard_tensors.h>


#include <algorithm>
#include <iostream>


#include "SolidSerial.hpp"


using namespace dealii;


namespace FE
{

template <int dim>
SolidSerial<dim>::SolidSerial(Triangulation<dim> &tria,
							  const std::string &fe_parameter_file,
							  const std::string &material_parameter_file)
:
parameters(fe_parameter_file),
triangulation(&tria),
fe(FESystem<dim>(FE_Q<dim>(parameters.poly_degree),dim)),
dof_handler(*triangulation),
u_fe(0),
u_x_fe(0),
u_y_fe(1),
u_z_fe(2),
quadrature(parameters.poly_degree+1),
face_quadrature(parameters.poly_degree+1),
material(material_parameter_file)
{
    Assert(dim==2 || dim==3,
           ExcMessage("This problem only works in 2 or 3 space dimensions."));
}


template <int dim>
void
SolidSerial<dim>::make_grid()
{
	GridGenerator::hyper_rectangle(*triangulation,
								   (dim == 3 ? Point<dim>(0.0, 0.0, 0.0) : Point<dim>(0.0, 0.0)),
								   (dim == 3 ? Point<dim>(1.0, 1.0, 1.0) : Point<dim>(1.0, 1.0)),
								   true);

    GridTools::scale(parameters.scale,*triangulation);

    // Since we wish to apply a Neumann BC to a patch on the top surface, we
    // must find the cell faces in this part of the domain and mark them with
    // a distinct boundary ID number.  The faces we are looking for are on the
    // top surface and will get boundary ID 17 (zero through five are already
    // used when creating the six faces of the cube domain):
    const double tol = 1e-6;

    for (const auto &cell : triangulation->active_cell_iterators())
    	for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
    		if (cell->face(face)->at_boundary())
    		{
    			if (dim==3)
    			{
                    if (std::abs(cell->face(face)->center()[2] - 1.0 * parameters.scale) < tol)
                      cell->face(face)->set_boundary_id(17);
    			}
    			else
    			{
                    if (std::abs(cell->face(face)->center()[1] - 1.0 * parameters.scale) < tol)
                      cell->face(face)->set_boundary_id(17);
    			}
    		}

    triangulation->refine_global(parameters.global_refinement);

    std::ofstream out("./mesh.vtu");

    GridOut().write_vtu(*triangulation,out);
}


template <int dim>
void
SolidSerial<dim>::system_setup()
{
	dof_handler.distribute_dofs(fe);

	DoFRenumbering::Cuthill_McKee(dof_handler);

	AffineConstraints<double> constraints;

	make_constraints(constraints);

	DynamicSparsityPattern dsp(dof_handler.n_dofs(),dof_handler.n_dofs());

	DoFTools::make_sparsity_pattern(dof_handler,
									dsp,
									constraints,
									/*keep_constrained_dofs*/ false);

	sparsity_pattern.copy_from(dsp);
}


template <int dim>
void
SolidSerial<dim>::make_constraints(AffineConstraints<double> &constraints,
		  	  	  	  	  	  	   const unsigned int /*it_nr*/) const
{
	constraints.clear();

	DoFTools::make_hanging_node_constraints(dof_handler,constraints);

	if (dim==3)
	{
		// On the face (bid=0) with normal in x-direction and at x=0 the
		// displacement in x-direction is homogeneously constrained: symmetry
		VectorTools::interpolate_boundary_values(dof_handler,
												 0,
												 ZeroFunction<dim>(dim),
		                                         constraints,
												 fe.component_mask(u_x_fe));

		// On the face (bid=2) with normal in y-direction and at y=0 the
		// displacement in y-direction is homogeneously constrained: symmetry
		VectorTools::interpolate_boundary_values(dof_handler,
												 2,
												 ZeroFunction<dim>(dim),
		                                         constraints,
												 fe.component_mask(u_y_fe));

		// On the bottom face (bid=4) displacement in z-direction is homogeneously constrained
		VectorTools::interpolate_boundary_values(dof_handler,
												 4,
												 ZeroFunction<dim>(dim),
		                                         constraints,
												 fe.component_mask(u_z_fe));
	}
	else
	{
		// On the bottom face (bid=2) displacement in y-direction is homogeneously constrained
		VectorTools::interpolate_boundary_values(dof_handler,
												 2,
												 ZeroFunction<dim>(dim),
		                                         constraints,
												 fe.component_mask(u_y_fe));

		// On the face (bid=0) with normal in x-direction and at x=0 the
		// displacement in x-direction is homogeneously constrained: symmetry
		VectorTools::interpolate_boundary_values(dof_handler,
												 0,
												 ZeroFunction<dim>(dim),
		                                         constraints,
												 fe.component_mask(u_x_fe));
	}
	constraints.close();
}


template <int dim>
void
SolidSerial<dim>::setup_qph(CellDataStorage<typename DoFHandler<dim>::cell_iterator,
							   	   	   	    QPData<dim>> &qp_data) const
{
	qp_data.initialize(dof_handler.begin_active(),
					   dof_handler.end(),
                       quadrature.size());

    // Next we setup the initial quadrature point data.
    // Note that when the quadrature point data is retrieved,
    // it is returned as a vector of smart pointers.
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
    	std::vector<std::shared_ptr<QPData<dim>>> qp_data_cell = qp_data.get_data(cell);

        Assert(qp_data_cell.size() == quadrature.size(),ExcInternalError());

        for (unsigned int qp = 0; qp<quadrature.size(); ++qp)
        	qp_data_cell[qp]->reinit(material);
    }
}


template <int dim>
bool
SolidSerial<dim>::update_qph(CellDataStorage<typename DoFHandler<dim>::cell_iterator,
							 	 	 	 	 QPData<dim>> &qp_data,
							 const Vector<double> &solution) const
{
	FEValues<dim> fe_v(fe,
					   quadrature,
					   update_gradients);

	std::vector<Tensor<2,dim>> qp_deformation_gradients(quadrature.size());

	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		fe_v.reinit(cell);

	    std::vector<std::shared_ptr<QPData<dim>>> qp_data_cell = qp_data.get_data(cell);

	    Assert(qp_data_cell.size() == quadrature.size(),ExcInternalError());

	    fe_v[u_fe].get_function_gradients(solution,qp_deformation_gradients);

	    for (unsigned int qp=0; qp<quadrature.size(); ++qp)
	    {
	    	const bool valid_det_F = qp_data_cell[qp]->update_values(qp_deformation_gradients[qp]);

	    	if (!valid_det_F)
	    		return false;
	    }
	}
	return true;
}


template <int dim>
void
SolidSerial<dim>::assemble_matrix(SparseMatrix<double> &matrix,
		 	 	 	 	 	 	  CellDataStorage<typename DoFHandler<dim>::cell_iterator,
								  	  	  	  	  QPData<dim>> &qp_data,
								  const AffineConstraints<double> &constraints) const
{
	matrix = 0;

	FEValues<dim> fe_v(fe,
					   quadrature,
					   update_gradients | update_quadrature_points | update_JxW_values);

	FullMatrix<double> cell_matrix(fe.dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

	SymmetricTensor<2,dim> var_C, delta_C, delta_var_C, tmp_product;

	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		fe_v.reinit(cell);

		cell->get_dof_indices(local_dof_indices);

		cell_matrix = 0;

	    std::vector<std::shared_ptr<QPData<dim>>> qp_data_cell = qp_data.get_data(cell);

	    Assert(qp_data_cell.size() == quadrature.size(),ExcInternalError());

	    for (unsigned int qp=0; qp<quadrature.size(); ++qp)
	    	for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
	    	{
	    		var_C = symmetrize(transpose(qp_data_cell[qp]->tensor_F) * fe_v[u_fe].gradient(i,qp));

	    		tmp_product = var_C * qp_data_cell[qp]->tangent;

	    		for (unsigned int j=0; j<fe.dofs_per_cell; ++j)
	    		{
	    			delta_C = symmetrize(transpose(qp_data_cell[qp]->tensor_F) * fe_v[u_fe].gradient(j,qp));

	    			delta_var_C = symmetrize(transpose(fe_v[u_fe].gradient(i,qp)) * fe_v[u_fe].gradient(j,qp));

	    			cell_matrix(i,j) += (tmp_product * delta_C) * fe_v.JxW(qp);

	    			cell_matrix(i,j) += (qp_data_cell[qp]->tensor_S * delta_var_C) * fe_v.JxW(qp);
	    		}
	    	}

	    constraints.distribute_local_to_global(cell_matrix,local_dof_indices,matrix);
	}
}


template <int dim>
void
SolidSerial<dim>::assemble_residuum(Vector<double> &residuum,
		 	 	 	   	   	   	    CellDataStorage<typename DoFHandler<dim>::cell_iterator,
		 	 	 	 	 	   	   	   	   	 	 	QPData<dim>> &qp_data,
									std::shared_ptr<Load<dim>> load,
									const AffineConstraints<double> &constraints) const
{
	// as the system K delta_u = -R has to be solved,
	// the residuum is multiplied with -1

	residuum = 0;

	FEValues<dim> fe_v(fe,
					   quadrature,
					   update_gradients | update_JxW_values);

	FEFaceValues<dim> fe_face_v(fe,
								face_quadrature,
								update_values | update_quadrature_points | update_JxW_values);

	Vector<double> cell_residuum(fe.dofs_per_cell);

	Vector<double> cell_rhs(fe.dofs_per_cell);

	std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

	std::vector<Point<dim>> fq_points(face_quadrature.size());

	std::vector<Tensor<1,dim>> load_fqp(face_quadrature.size());

	SymmetricTensor<2,dim> var_C;

	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		fe_v.reinit(cell);

		cell->get_dof_indices(local_dof_indices);

		cell_residuum = 0;

	    std::vector<std::shared_ptr<QPData<dim>>> qp_data_cell = qp_data.get_data(cell);

	    Assert(qp_data_cell.size() == quadrature.size(),ExcInternalError());

	    for (unsigned int qp=0; qp<quadrature.size(); ++qp)
	    {
	    	for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
	    	{
	    		var_C = symmetrize(transpose(qp_data_cell[qp]->tensor_F) * fe_v[u_fe].gradient(i,qp));

	    		cell_residuum[i] -= (qp_data_cell[qp]->tensor_S * var_C) * fe_v.JxW(qp);
	    	}
	    }
	    if (cell->at_boundary())
	    	for (unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
	    		if (cell->face(face)->boundary_id() == 17)
	    		{
	    			fe_face_v.reinit(cell,face);

	    			fq_points = fe_face_v.get_quadrature_points();

	    			load->tensor_value_list(fq_points,load_fqp);

	    			for (unsigned int fqp=0; fqp<face_quadrature.size(); ++fqp)
	    				for (unsigned int i=0; i<fe.dofs_per_cell; ++i)
	    					cell_residuum[i] += (load_fqp[fqp] * fe_face_v[u_fe].value(i,fqp)) * fe_face_v.JxW(fqp);
	    		}

	    constraints.distribute_local_to_global(cell_residuum,local_dof_indices,residuum);
	}
}


template <int dim>
bool
SolidSerial<dim>::solve_NewtonRaphson(Vector<double> &solution,
									  CellDataStorage<typename DoFHandler<dim>::cell_iterator,
				 	 	 	 	 	   	   	   	   	  QPData<dim>> &qp_data,
									  std::shared_ptr<Load<dim>> load) const
{
	Vector<double> solution_increment(solution.size());

	AffineConstraints<double> constraints;

	make_constraints(constraints,0);


	update_qph(qp_data,solution);

	Vector<double> residuum(solution.size());

	assemble_residuum(residuum,qp_data,load,constraints);

	SparseMatrix<double> matrix(sparsity_pattern);

	assemble_matrix(matrix,qp_data,constraints);

	const double initial_residuum = residuum.l2_norm();

	std::cout << "      Norm of initial residuum: " << initial_residuum << std::endl;

    for (unsigned int it=0; it< parameters.max_iterations_NR; ++it)
    {
    	 solve_linear_system(solution_increment,matrix,residuum);

    	 constraints.distribute(solution_increment);

    	 solution += solution_increment;

    	 const bool success = update_qph(qp_data,solution);

    	 if (!success)
    	 {
    		 std::cout << "      Negative det_F occured ==> reset and decrease load " << std::endl;

    		 return false;
    	 }
    	 assemble_residuum(residuum,qp_data,load,constraints);

    	 const double norm_residuum = residuum.l2_norm();

    	 std::cout << "      Norm of residuum after iteration " << it+1 << ": " << norm_residuum << std::endl;

    	 if (norm_residuum / initial_residuum < parameters.tol_f)
    		 return true;

    	 const double norm_increment = solution_increment.l2_norm();

    	 if ((it>0) && (norm_increment < parameters.tol_u))
    		 return true;

    	assemble_matrix(matrix,qp_data,constraints);
    }

    return false;
}


template <int dim>
std::pair<unsigned int,double>
SolidSerial<dim>::solve_linear_system(Vector<double> &solution,
									  const SparseMatrix<double> &matrix,
									  const Vector<double> &rhs) const
{
	std::pair<unsigned int,double> ret;

    if (parameters.type_lin == "CG")
    {
        const auto solver_its =
        	static_cast<unsigned int>(matrix.m() * parameters.max_iterations_lin);

        const double tol_sol = parameters.tol_lin * rhs.l2_norm();

        SolverControl solver_control(solver_its,tol_sol);

        GrowingVectorMemory<Vector<double>> GVM;

        SolverCG<Vector<double>> solver_CG(solver_control, GVM);

        PreconditionSelector<SparseMatrix<double>,Vector<double>>
          preconditioner(parameters.preconditioner_type,
                         parameters.preconditioner_relaxation);

        preconditioner.use_matrix(matrix);

        solver_CG.solve(matrix,
        				solution,
                        rhs,
                        preconditioner);

        ret.first = solver_control.last_step();

        ret.second = solver_control.last_value();
    }
    else if (parameters.type_lin == "Direct")
    {
        SparseDirectUMFPACK A_direct;

        A_direct.initialize(matrix);

        A_direct.vmult(solution,rhs);

        ret.first = 1;

        ret.second = 0.0;
    }
    return ret;
}


template <int dim>
void
SolidSerial<dim>::output_results(const Vector<double> &vec,
								 CellDataStorage<typename DoFHandler<dim>::cell_iterator,QPData<dim>> &qp_data,
								 const std::string &file_name) const
{
	DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	data_component_interpretation(dim,
								  DataComponentInterpretation::component_is_part_of_vector);

	std::vector<std::string> solution_name(dim,"displacement");

	Vector<double> norm_of_stress(triangulation->n_active_cells());

	SymmetricTensor<2, dim> accumulated_stress;

	for (const auto &cell : dof_handler.active_cell_iterators())
	{
		std::vector<std::shared_ptr<QPData<dim>>> qp_data_cell = qp_data.get_data(cell);

		for (unsigned int qp=0; qp<quadrature.size(); ++qp)
		{
			accumulated_stress += qp_data_cell[qp]->tensor_S;

			norm_of_stress(cell->active_cell_index()) = (accumulated_stress / quadrature.size()).norm();
		}
	}
	data_out.add_data_vector(norm_of_stress, "norm_of_stress");

	data_out.add_data_vector(vec,
							 solution_name,
							 DataOut<dim>::type_dof_data,
							 data_component_interpretation);

	data_out.build_patches();

    std::ofstream output(file_name);

    data_out.write_vtu(output);
}


template <int dim>
void
SolidSerial<dim>::output_results(const Vector<double> &vec,
								 const std::string &file_name) const
{
	DataOut<dim> data_out;

	data_out.attach_dof_handler(dof_handler);

	std::vector<DataComponentInterpretation::DataComponentInterpretation>
	data_component_interpretation(dim,
								  DataComponentInterpretation::component_is_part_of_vector);

	std::vector<std::string> solution_name(dim,"d");

	data_out.add_data_vector(vec,
							 solution_name,
							 DataOut<dim>::type_dof_data,
							 data_component_interpretation);

	data_out.build_patches();

    std::ofstream output(file_name);

    data_out.write_vtu(output);
}


template <int dim>
void
SolidSerial<dim>::run()
{
	make_grid();

	system_setup();

	Point<dim> origin_load;
	if (dim==3)
		origin_load[2] = 1;
	else
		origin_load[1] = 1;

	std::shared_ptr<Load<dim>> load =
			std::make_shared<ParabolicLoad<dim>>(origin_load,
												 parameters.p_0 * parameters.scale * parameters.scale,
												 parameters.parabola_root);

	std::cout << std::endl << std::endl;

	std::cout << "Triangulation consists of " << triangulation->n_active_cells() << " elements"
			  << std::endl
			  << "Number of DoFs: " << dof_handler.n_dofs()
			  << std::endl << std::endl;

	Vector<double> solution(dof_handler.n_dofs()), solution_reset(dof_handler.n_dofs());

	CellDataStorage<typename DoFHandler<dim>::cell_iterator,QPData<dim>> qp_data;

	setup_qph(qp_data);

	const double initial_load_factor = parameters.initial_load_factor;

	double load_factor_old = 0;

	double load_factor_increment = initial_load_factor;

	double load_factor = load_factor_old + load_factor_increment;

	for (unsigned int count=0; count < parameters.max_n_load_steps; ++count)
	{
		std::cout << "   Solve non-linear problem for load factor = " << load_factor << std::endl;

		load->set_load_factor(load_factor);

		const bool conv = solve_NewtonRaphson(solution,
											  qp_data,
											  load);

		if (conv)
		{
			output_results(solution,qp_data,"./solution_"+Utilities::to_string(count+1)+".vtu");

			// if convergence is obtained for load_factor=1, the problem is solved
			if (load_factor == 1)
				return;

			solution_reset = solution;

			load_factor_old = load_factor;

			load_factor += load_factor_increment;

			if (load_factor > 1)
			{
				load_factor = 1;
				load_factor_increment = load_factor - load_factor_old;
			}
		}
		// no convergence obtained
		else
		{
			if (count < parameters.max_n_load_steps-1)
			{
				// reset solution
				solution = solution_reset;

				load_factor_increment *= 0.5;

				load_factor = load_factor_old + load_factor_increment;
			}
			else
			{
				solution = 0;

				std::cout << "   Non-linear problem could not be solved within the maximum number of load steps = "
						  << parameters.max_n_load_steps
						  << " and stopped at load factor "
						  << load_factor << std::endl;

				return;
			}
		}
	}
}


}//namespace FE


#include "SolidSerial.inst"
