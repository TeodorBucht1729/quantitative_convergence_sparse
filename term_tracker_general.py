from fractions import Fraction
from itertools import product, permutations, chain, combinations
import math
import os
import pickle
from typing import Self
import copy
import networkx as nx
import numpy as np
import sys
import gaussian_elim
import more_itertools as mit
from scipy.sparse import csc_matrix, coo_matrix
from scipy.sparse.linalg import lsqr


global use_random_edge
global throw_away_higher_derivatives
global permutation_groups_global
global q_cutoff
global latex_print_outs
global alp

alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

use_random_edge = True
throw_away_higher_derivatives = False
permutation_groups_global = None
latex_print_outs = True
q_cutoff = 3

h_index_offset = -1000
exp_variable_index = 1001
random_edge_index = 1002


class CumulantPoly:
    """
    Class representing a multivariable polynomial with commuting variables. 
    """

    def __init__(self, term, coeff):
        """
        Initializes a monomialx_1^p_1 x^2^p_2 ... x_n^p_n
        term: ((0, 0), (1, p_1), (2, p_2), ..., (n, p_n))
        coeff: float
        """

        # add (0, 0) to term if not present, and make sure it is sorted
        term_list = list(term)
        if (0, 0) not in term_list:
            term_list.append((0, 0))
        term_list.sort()
        term = tuple(term_list)
        self.poly = {term : coeff}
        # always add a scalar term: (and set it to zero if needed)
        if ((0,0),) not in self.poly:
            self.poly[((0,0),)] = 0

    def __add__(self, other_poly: Self):

        all_keys = list(self.poly.keys())
        all_keys.extend(list(other_poly.poly.keys()))
        
        all_keys = set(all_keys)
        for key in all_keys:
            if key not in other_poly.poly:
                continue
            if key in self.poly:
                self.poly[key] += other_poly.poly[key]
            else:
                self.poly[key] = other_poly.poly[key]

    def is_zero(self):
        for term in self.poly:
            if self.poly[term] != 0:
                return False
        return True
    
    def get_coeff_of_term(self, term):
        if term in self.poly:
            return self.poly[term]
        return 0.0

    def is_constant(self):
        if len(self.poly.keys()) == 1:
            return True 

    def __mul__(self, other_poly: Self):
        result_poly = CumulantPoly(((0, 0),), 0)
        for key1 in self.poly:
            for key2 in other_poly.poly:
                new_coeff = self.poly[key1] * other_poly.poly[key2]
                dict1 = dict(key1)
                dict2 = dict(key2)

                all_factors = list(dict1.keys())
                all_factors.extend(list(dict2.keys()))
                all_factors = set(all_factors)
                for factor in all_factors:
                    if factor not in dict2:
                        continue
                    if factor in dict1:
                        dict1[factor] += dict2[factor]
                    else:
                        dict1[factor] = dict2[factor]
                new_key = tuple(sorted(list(dict1.items())))
                result_poly + CumulantPoly(new_key, new_coeff)
        self.poly = result_poly.poly

    def __str__(self) -> str:
        if self.is_zero():
            return "0"
        terms = []
        for term in self.poly:
            if self.poly[term] == 0:
                continue
            if term == ((0,0), (4,1), (exp_variable_index, 1)):
                cumulant_string = "e^{-t}(1 - e^{-t}) \\kappa_4"
            else:
                cumulant_string = ''.join([f"\kappa_{ind}^{p}" if ind != 0 else '1' for (ind, p) in term])
            terms.append(f"{Fraction(self.poly[term])} {cumulant_string}")
        terms.sort()
        return " + ".join(terms)


class TermCollection:
    """
    Represents a sum of standard terms of the form 
    N^(N_deg)/(N^(#I) q^(q_deg)) sum_(I) cumulant_poly E[F^(i_0)(X(t)) prod_{i=1}^{i_0} ΔIm prod_{l=1}^{n_i} G_(x_l^{(i)} y_l^{(i)})].
    The terms are saved in the dict self.terms, the keys are of the form (q_deg, N_deg, F_derivative, tuple_of_prods, h_list),
    an example key is (2, 1, 1, ((((0, 2), 2), ((1, 1), 2), ((2, 2), 1)),), ()), which represents the term
    N/q^2 sum_(a,b,c) E[F'(X(t)) ΔIm G_ac^2 G_bb^2 G_cc]. The values in self.terms are Term-objects.
    """

    def __init__(self): 
        self.terms = {}

    def __add__(self, other_term_collection: Self):
        for term in self.terms:
            if term in other_term_collection.terms:
                self.terms[term] + other_term_collection.terms[term]
        for term in other_term_collection.terms:
            if term not in self.terms:
                self.terms[term] = other_term_collection.terms[term]

    def mult_with_constant(self, constant: CumulantPoly):
        for term_key in self.terms:
            self.terms[term_key].coeff * constant

    def filter_trivial_zero_terms(self):
        term_keys = copy.deepcopy(list(self.terms.keys()))
        for term_key in term_keys:
            if self.terms[term_key].coeff.is_zero():
                del self.terms[term_key]
                continue
            for outer_product in self.terms[term_key].prod_dict:
                if len(outer_product) == 0:
                    del self.terms[term_key]
                    break


    def save_instance(self, save_name):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, f"{save_name}.pickle"), "wb") as f:
            pickle.dump(self, f)

    def filter_and_group_equivalent(self, total_degree_filter, q_degree_filter, degree_filter, use_pre_saved_permutation_groups = True):
        self.filter_high_degree_terms(total_degree_filter)
        self.filter_high_q_degree_terms(q_degree_filter)
        self.filter_trivial_zero_terms()
        self.filter_degree_terms(degree_filter)
        self.group_equivalent(use_pre_saved_permutation_groups=use_pre_saved_permutation_groups)

    def filter_degree_terms(self, degree_limit):
        term_keys = copy.deepcopy(tuple(self.terms.keys()))
        for term_key in term_keys:
            if self.terms[term_key].get_degree() >= degree_limit:
                del self.terms[term_key]

    def filter_high_degree_terms(self, degree_limit):
        """
        deletes all terms for which degree + q_deg >= degree_limit
        """
        term_keys = copy.deepcopy(tuple(self.terms.keys()))
        for term_key in term_keys:
            if self.terms[term_key].get_degree() + self.terms[term_key].q_deg >= degree_limit:
                del self.terms[term_key]

    def filter_high_q_degree_terms(self, degree_limit):
        """
        deletes all terms for which q_deg >= degree_limit
        """
        term_keys = copy.deepcopy(tuple(self.terms.keys()))
        for term_key in term_keys:
            if self.terms[term_key].q_deg >= degree_limit:
                del self.terms[term_key]

    def group_equivalent(self, use_pre_saved_permutation_groups = True):
        """ 
        Join terms that are equivalent up to permutation of indices, this is not described in the article as it is not needed for the proof,
        but it makes intermediate results more easy to view.  
        """
        global permutation_groups_global

        if use_pre_saved_permutation_groups:
            if permutation_groups_global is None: 
                dir_path = os.path.dirname(os.path.realpath(__file__))
                if "saved_permutation_groups_general.pickle" in os.listdir(dir_path):
                    with open(os.path.join(dir_path, "saved_permutation_groups_general.pickle"), "rb") as f:
                        permutation_groups = pickle.load(f)
                else:
                    permutation_groups = {}
            else:
                permutation_groups = permutation_groups_global
        else:
            permutation_groups = {}

        term_keys = copy.deepcopy(tuple(self.terms.keys()))
        for term in term_keys:
            if term not in self.terms:
                # term has already been grouped
                continue

            eq_perm_term_found = False
            found_perm_term = None
            if term in permutation_groups:
                eq_perm_term_found = True
                found_perm_term = term
            else:
                for perm_key in permutation_groups:
                    term1 = generate_term_obj_from_key(perm_key)
                    term2 = generate_term_obj_from_key(term)
                    if two_terms_equivalent(term1, term2):
                        eq_perm_term_found = True
                        found_perm_term = tuple(perm_key)
                        permutation_groups[term] = permutation_groups[found_perm_term]
                        break

            if eq_perm_term_found:
                # add term to what it is pointing to
                group_representative = permutation_groups[found_perm_term]
                if group_representative != term:
                    self.terms[term].prod_dict = [dict(outer_factor) for outer_factor in group_representative[3]]
                    if group_representative in self.terms:
                        self.terms[group_representative] + self.terms[term]
                        del self.terms[term]
                    else:
                        self.terms[group_representative] = self.terms[term]
                        del self.terms[term]
                continue

            # is this place is reached, the term has not occured previously in the permutation groups. 
            found_equivalent_terms = []
            existing_group_found = False
            existing_group_representative = None

            for other_term in self.terms:
                if two_terms_equivalent(self.terms[term], self.terms[other_term]):
                    # if term != other_term:
                        # print("\nFound equivalent terms\n", self.terms[term], "and\n", self.terms[other_term], "\n\n")
                    found_equivalent_terms.append(other_term)
                    if other_term in permutation_groups:
                        existing_group_found = True
                        existing_group_representative = permutation_groups[other_term]

            # remove duplicates
            found_equivalent_terms = set(found_equivalent_terms)

            # add all equivalent terms to the same Term-object
            #TODO: possibly rewrite the following code, it is not very pretty, but it works
            if existing_group_found:
                for term_key in found_equivalent_terms:
                    permutation_groups[term_key] = existing_group_representative
                    if term_key == existing_group_representative:
                        continue
                    # add term_key to existing_group_representative
                    self.terms[term_key].prod_dict = [dict(outer_factor) for outer_factor in existing_group_representative[3]]
                    if existing_group_representative not in self.terms:
                        self.terms[existing_group_representative] = Term(self.terms[term_key].F_derivative, self.terms[term_key].prod_dict, self.terms[term_key].q_deg, self.terms[term_key].N_deg, CumulantPoly(((0,0),), 0))
                    self.terms[existing_group_representative] + self.terms[term_key]
                    del self.terms[term_key]
            else:
                for term_key in found_equivalent_terms:
                    permutation_groups[term_key] = term
                    if term_key == term:
                        continue
                    # add term_key to term
                    self.terms[term_key].prod_dict = [dict(outer_factor) for outer_factor in term[3]]
                    self.terms[term] + self.terms[term_key]
                    del self.terms[term_key]

            if term in self.terms and self.terms[term].coeff.is_zero():
                del self.terms[term]
        
        # save permutation groups which may have been updated
        # with open(os.path.join(dir_path, "saved_permutation_groups_general.pickle"), "wb") as f:
        #     pickle.dump(permutation_groups, f)

    def rearrange_variables_all_terms(self):
        term_keys = copy.deepcopy(tuple(self.terms.keys()))
        for term_key in term_keys:
            self.terms[term_key].rearrange_vars()
            if term_key != self.terms[term_key].get_key():
                self.add_term(self.terms[term_key])
                del self.terms[term_key]


    def add_m_with_z_rule(self, term_to_expand, outer_factor_to_expand = None, reuse_index = None, print_out = False):

        term = self.terms[term_to_expand]

        if reuse_index is None:
            v = term.get_fresh_index(0)
        else:
            v = reuse_index
        j = term.get_fresh_index(v+1)

        outer_factor_ind = 0
        if outer_factor_to_expand is not None:
            dict_outer_factor_to_expand = dict(outer_factor_to_expand)
            while(term.prod_dict[outer_factor_ind] != dict_outer_factor_to_expand):
                outer_factor_ind += 1
            assert term.prod_dict[outer_factor_ind] == dict_outer_factor_to_expand, "outer_factor_to_expand does not exist in the specified term"

        if print_out:
            alp = ['v', 'a', 'b', 'c', 'j', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z']
            print(f"Expanding 1 with {alp[v]} and {alp[j]} in", str(term), f", \end{{equation*}} \nto obtain ")

        new_prod_dict_1 = copy.deepcopy(term.prod_dict)
        
        add_to_dict(new_prod_dict_1[outer_factor_ind], ordind(v, v), 1)

        new_coefficient_1 = CumulantPoly(((0, 0),), -2.0)
        new_coefficient_1 * term.coeff

        new_term_1 = Term(term.F_derivative, new_prod_dict_1, term.q_deg, term.N_deg, new_coefficient_1)
        new_term_1.rearrange_vars()
        self.add_term(new_term_1)

        if print_out:
            print(new_term_1)
            print(new_term_1.get_key())

        higher_order_contribution_term = copy.deepcopy(new_term_1)
        higher_order_contribution_term.q_deg += 2
        higher_order_contribution_term.coeff * CumulantPoly(((0, 0),(201, 1)), 1.0)
        self.add_term(higher_order_contribution_term)

        new_prod_dict_2 = copy.deepcopy(term.prod_dict)
        add_to_dict(new_prod_dict_2[outer_factor_ind], ordind(v, j), 1)

        new_coefficient_2 = CumulantPoly(((0, 0),), 1.0)
        new_coefficient_2 * term.coeff    

        new_term_2 = Term(term.F_derivative, new_prod_dict_2, term.q_deg, term.N_deg+1, new_coefficient_2)
        new_term_2.add_h(ordind(j, v))
        new_term_2.rearrange_vars()

        # print("\\begin{equation*}", str(new_term), ". \end{equation*}")
        
        new_term_2.N_deg -= 1
        cumulant_expansion = new_term_2.expand_hs()

        if print_out:
            cumulant_expansion.print_terms_in_order()
            print("\n\n")


        del self.terms[term_to_expand] # remove the expanded term

        self + cumulant_expansion

    def expand_off_diagonal_with_z_rule(self, term_to_expand, outer_factor_to_expand, factor_to_expand, change_index_order = False):

        assert factor_to_expand[0] != factor_to_expand[1], "Can only be used on off-diagonal factors"

        term = self.terms[term_to_expand]

        used_indices = set(term.get_var_indices())

        def get_fresh_index(used_variables, latest_var):
            while latest_var in used_variables:
                latest_var += 1
            used_variables.add(latest_var)
            return latest_var

        outer_factor_ind = 0
        dict_outer_factor_to_expand = dict(outer_factor_to_expand)
        while(term.prod_dict[outer_factor_ind] != dict_outer_factor_to_expand):
            outer_factor_ind += 1
        assert term.prod_dict[outer_factor_ind] == dict_outer_factor_to_expand, "outer_factor_to_expand does not exist in the specified term"

        fresh_index = 0
        fresh_index = get_fresh_index(used_indices, fresh_index)
        j = fresh_index
        v = factor_to_expand[0]
        a = factor_to_expand[1]

        alp = ['v', 'a', 'b', 'c', 'j', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z']
        # print(f"Expanding $G_{{{alp[v]}{alp[a]}}}$ in \\begin{{equation*}}", str(term), f", \end{{equation*}} \nto obtain ")

        new_prod_dict = copy.deepcopy(term.prod_dict)

        # prepare to add h_s 
        add_to_dict(new_prod_dict[outer_factor_ind], ordind(v, a), -1)
        if change_index_order:
            add_to_dict(new_prod_dict[outer_factor_ind], ordind(a, j), 1)
        else:
            add_to_dict(new_prod_dict[outer_factor_ind], ordind(v, j), 1)

        new_coefficient = CumulantPoly(((0, 0),), 0.5)
        new_coefficient * term.coeff

        new_term = Term(term.F_derivative, new_prod_dict, term.q_deg, term.N_deg+1, new_coefficient)
        if change_index_order:
            new_term.add_h(ordind(v, j))
        else:
            new_term.add_h(ordind(a, j))
        # print("\\begin{equation*}", str(new_term), ". \end{equation*}")
        new_term.rearrange_vars()
        new_term.N_deg -= 1
        cumulant_expansion = new_term.expand_hs()

        higher_order_contribution_term = copy.deepcopy(term)
        higher_order_contribution_term.q_deg += 2
        higher_order_contribution_term.coeff * CumulantPoly(((0, 0),(201, 1)), -1.0)
        higher_order_contribution_term.rearrange_vars()
        self.add_term(higher_order_contribution_term)

        del self.terms[term_to_expand] # remove the expanded term

        self + cumulant_expansion


    def add_term(self, term):
        key = term.get_key()
        if key in self.terms:
            self.terms[key] + term
        else:
            self.terms[key] = term
            
        # remove zero terms to save memory
        if self.terms[key].coeff.is_zero():
            del self.terms[key]

    def print_terms_in_order(self, degree_limit = 100000, print_keys=True):
        global latex_print_outs
        term_entries = list(self.terms.items())
        term_entries.sort(key=lambda x: (x[1].q_deg, x[1].get_degree()))
        print("\n\n\nTerm Collection:\n\n\n")
        for term_entry in term_entries:
            if term_entry[1].get_degree() + term_entry[1].q_deg >= degree_limit:
                continue
            if latex_print_outs:
                print("&", term_entry[1], "\\\\")
            else:
                print(term_entry[1])
            if print_keys:
                print("Key:", term_entry[0])

def load_term_collection(save_name) -> TermCollection:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if f"{save_name}.pickle" in os.listdir(dir_path):
        print("Loading", os.path.join(dir_path, f"{save_name}.pickle"))
        with open(os.path.join(dir_path, f"{save_name}.pickle"), "rb") as f:
            term_collection = pickle.load(f)
    return term_collection

def ordind(a, b):
    return (min(a,b), max(a,b))

def add_to_dict(dic, key, val):
    if key in dic:
        dic[key] += val
    else:
        dic[key] = val
    if dic[key] == 0:
        del dic[key]

class Term:
    """
    Represents a term of the form 
    N^(N_deg)/(N^(#I) q^(q_deg)) sum_(I) cumulant_poly E[F^(i_0)(X(t)) prod_{i=1}^{i_0} ΔIm prod_{l=1}^{n_i} G_(x_l^{(i)} y_l^{(i)})]
    """

    def __init__(self, F_derivate: int, prod_dict: list[dict[tuple, int]], q_deg: int, N_deg: int, coeff: CumulantPoly):

        self.F_derivative = F_derivate
        self.prod_dict = prod_dict # now needs to be a list of dicts? 
        self.q_deg = q_deg
        self.N_deg = N_deg 
        self.coeff = coeff
        self.h_list = [] 

    def __add__(self, other_term):
        assert self.get_key() == other_term.get_key(), "terms with dirrefent keys can't be added"
        self.coeff + other_term.coeff

    @staticmethod
    def latex_string_from_outer_factor(outer_factor):
        global alp
        G_strings = []
        if len(outer_factor) > 0:
            G_strings.append("\Dim (")
            G_strings.append("".join([f"G_{{{alp[f[0][0]]}{alp[f[0][1]]}}}{f'^{{{f[1]}}}' if f[1] != 1 else ''}" for f in outer_factor]))
            G_strings.append(")")
        else:
            G_strings.append("\Dim (1)")
        return "".join(G_strings)

    def __str__(self):
        global latex_print_outs
        global alp
        # alp = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        # ['g', 'h', 'i', 'k', 'l', 'm', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z']
        if not latex_print_outs:
            key = self.get_key()
            G_strings = []
            for outer_factor in key[3]:
                G_strings.append("∆Im(")
                G_strings.append("".join([f"G_{alp[f[0][0]]}{alp[f[0][1]]}^{f[1]}" for f in outer_factor]))
                G_strings.append(")")
            h_strings = [f"h_{alp[h[0]]}{alp[h[1]]}" for h in self.h_list]
            return f"N^{self.N_deg}/q^{self.q_deg} {str(self.coeff)} E[F^({self.F_derivative})(X){''.join(h_strings)}{''.join(G_strings)}], degree {self.get_degree()}"
        else:
            key = self.get_key()
            G_strings = []
            for outer_factor in key[3]:
                G_strings.append(Term.latex_string_from_outer_factor(outer_factor))
            h_strings = [f"h_{{{alp[h[0]]}{alp[h[1]]}}}" for h in self.h_list]
            nbr_vars = self.get_nbr_vars()
            if self.N_deg == 1:
                return f"{str(self.coeff)} \\frac{{N}}{{q^{self.q_deg}N^{nbr_vars}}}  \sum_{{{','.join(alp[:nbr_vars])}}} \EX \left[F^{{({self.F_derivative})}}(X(t)){''.join(h_strings)}{''.join(G_strings)}\\right]"
            else:
                return f"{str(self.coeff)} \\frac{{N^{self.N_deg}}}{{q^{self.q_deg}N^{nbr_vars}}} \sum_{{{','.join(alp[:nbr_vars])}}} \EX \left[F^{{({self.F_derivative})}}(X(t)){''.join(h_strings)}{''.join(G_strings)}\\right]"


    def get_var_indices(self):
        vars = set()
        for outer_factor in self.prod_dict:
            for factor in outer_factor:
                vars.add(factor[0])
                vars.add(factor[1])
        for h in self.h_list:
            vars.add(h[0])
            vars.add(h[1])
        return list(vars)

    def get_ms_indices(self):
        ms = []
        var_occurances = self.get_nbr_var_occurances()
        for outer_factor in self.prod_dict:
            for factor in outer_factor:
                if factor[0] == factor[1] and var_occurances[factor[0]] == 2:
                    ms.append(factor[0])
        return ms

    def get_nbr_vars(self):
        used_vars = self.get_var_indices()
        return len(used_vars)

    def get_nbr_Gs(self):
        var_occurances = self.get_nbr_var_occurances()
        total_occ = 0
        for var in var_occurances:
            total_occ += var_occurances[var]
        total_occ //= 2
        return total_occ

    def get_G_list(self):
        G_list = []
        for outer_factor in self.prod_dict:
            for factor in outer_factor:
                for _ in range(outer_factor[factor]):
                    G_list.append(factor)
        return G_list

    def get_nbr_var_occurances(self):
        var_occurances = {}
        for outer_factor in self.prod_dict:
            for factor in outer_factor:
                add_to_dict(var_occurances, factor[0], outer_factor[factor])
                add_to_dict(var_occurances, factor[1], outer_factor[factor])
        return var_occurances

    def get_fresh_index(self, start_index):
        ind = start_index
        used_variables = self.get_var_indices()
        while(ind in used_variables):
            ind += 1
        return ind

    def rearrange_vars(self):
        """
        Renames the variables to be 0, 1, ..., nbr_vars-1
        """
        used_vars = self.get_var_indices()
        used_vars = list(used_vars)
        used_vars.sort()

        if len(used_vars) > 0 and used_vars[-1] > len(used_vars)-1:
            # variables should be rearranged
            map_to_new_vars = {}
            for i in range(self.get_nbr_vars()):
                map_to_new_vars[used_vars[i]] = i
            new_prod_dict = []
            for outer_factor in self.prod_dict:
                new_inner_prod_dict = {}
                for factor in outer_factor:
                    new_inner_prod_dict[(map_to_new_vars[factor[0]], map_to_new_vars[factor[1]])] = outer_factor[factor]
                new_prod_dict.append(new_inner_prod_dict)
            for i in range(len(self.h_list)):
                self.h_list[i] = (map_to_new_vars[self.h_list[i][0]], map_to_new_vars[self.h_list[i][1]])
            self.prod_dict = new_prod_dict


    def get_key(self):
        product_list = []
        for outer_product in self.prod_dict:
            product_list.append(tuple(sorted(tuple(outer_product.items()))))
        product_list.sort()
        product_list = tuple(product_list)
        key = (self.q_deg, self.N_deg, self.F_derivative, product_list,  tuple(sorted(self.h_list)))
        return key
    
    def get_degree(self):
        ans = 0
        for outer_factor in self.prod_dict:
            for factor in outer_factor:
                if factor[0] != factor[1]:
                    ans += outer_factor[factor]
        return ans

    def add_h(self, h_ind):
        self.h_list.append(h_ind)


    def expand_hs(self, h_to_expand = None) -> TermCollection:
        """ 
        Perform cumulant expansion on all the h:s in self. Note: can one expand one h at a time. 
        """
        global use_random_edge
        global throw_away_higher_derivatives
        global q_cutoff
        n_h = len(self.h_list)

        if n_h == 0:
            one_term_term_collection = TermCollection()
            one_term_term_collection.add_term(copy.deepcopy(self))
            return one_term_term_collection

        if h_to_expand is None:
            h = self.h_list[0]
        else:
            assert h_to_expand in self.h_list, "that h missing from h_list"
            h = h_to_expand

        self.h_list.remove(h)
        assert h[0] <= h[1], "should be ordered"


        cumulant_orders = list(range(1, q_cutoff + 1 - self.q_deg + 1))
        # print("Cumulant powers:", cumulant_orders)
        result_term_collection = TermCollection()

        computed_curr_term_collections = {}

        for cumulant_order in cumulant_orders:

            curr_term_collection = TermCollection()
            curr_term_collection.add_term(self)

            # filter out unmatched terms by ignoring the terms with odd cumulants
            if cumulant_order % 2 == 1:
                continue

        
            for diff_ind in range(1, cumulant_order):
                # perform one differentiation of curr_term_collection w.r.t. h

                if diff_ind in computed_curr_term_collections:
                    curr_term_collection = copy.deepcopy(computed_curr_term_collections[diff_ind])
                    continue

                new_term_collection = TermCollection()

                for term_key in curr_term_collection.terms:
                    term_to_diff = curr_term_collection.terms[term_key]

                    if not throw_away_higher_derivatives:
                        # here a term should be added that hits the F(\Chi)
                        new_prod_dict = copy.deepcopy(term_to_diff.prod_dict)
                        new_prod_dict.append(dict())
                        add_to_dict(new_prod_dict[-1], ordind(h[0], h[1]), 1)
                        new_coefficient = CumulantPoly(((0, 0),), -2)
                        new_coefficient * term_to_diff.coeff
                        
                        new_term = Term(term_to_diff.F_derivative + 1, new_prod_dict, term_to_diff.q_deg, term_to_diff.N_deg, new_coefficient) 
                        new_h_list = copy.deepcopy(term_to_diff.h_list)
                        new_term.h_list = new_h_list

                        new_term_collection.add_term(new_term)
                        
                        # the contribution from the random shift when the derivative hits F(\Chi)
                        if use_random_edge:
                            v = term_to_diff.get_fresh_index(0)
                            new_prod_dict = copy.deepcopy(term_to_diff.prod_dict)
                            new_prod_dict.append(dict())
                            add_to_dict(new_prod_dict[-1], ordind(v, v), 1)
                            new_coefficient = CumulantPoly(((0, 0),), 4)
                            new_coefficient * term_to_diff.coeff
                            
                            new_term = Term(term_to_diff.F_derivative + 1, new_prod_dict, term_to_diff.q_deg, term_to_diff.N_deg, new_coefficient) 
                            new_h_list = copy.deepcopy(term_to_diff.h_list)
                            new_term.h_list = new_h_list
                            new_term.add_h(ordind(h[0], h[1]))

                            new_term_collection.add_term(new_term)

                    # the derivative hits a h
                    h_count = term_to_diff.h_list.count(h)
                    if h_count >= 1:
                        new_dict = copy.deepcopy(term_to_diff.prod_dict)
                        new_coefficient = CumulantPoly(((0, 0),), h_count)
                        new_coefficient * term_to_diff.coeff
                        
                        new_term = Term(term_to_diff.F_derivative, new_dict, term_to_diff.q_deg, term_to_diff.N_deg, new_coefficient)
                        new_h_list = copy.deepcopy(term_to_diff.h_list)
                        new_h_list.remove(h)
                        new_term.h_list = new_h_list
                        new_term_collection.add_term(new_term)
                        # print(new_term)

                    for outer_factor_ind, outer_factor in enumerate(term_to_diff.prod_dict):
                        # the derivative hits outer_factor here
                        for factor in outer_factor:

                            # first contribution
                            new_prod_dict = copy.deepcopy(term_to_diff.prod_dict)
                            
                            add_to_dict(new_prod_dict[outer_factor_ind], ordind(factor[0], h[0]), 1)
                            add_to_dict(new_prod_dict[outer_factor_ind], ordind(factor[1], h[1]), 1)
                            add_to_dict(new_prod_dict[outer_factor_ind], factor, -1)

                            new_coefficient = CumulantPoly(((0, 0),), -outer_factor[factor])
                            new_coefficient * term_to_diff.coeff
                            
                            new_term = Term(term_to_diff.F_derivative, new_prod_dict, term_to_diff.q_deg, term_to_diff.N_deg, new_coefficient)
                            new_h_list = copy.deepcopy(term_to_diff.h_list)
                            new_term.h_list = new_h_list
                            new_term_collection.add_term(new_term)

                            # second contribution
                            new_prod_dict = copy.deepcopy(term_to_diff.prod_dict)
                            add_to_dict(new_prod_dict[outer_factor_ind], ordind(factor[0], h[1]), 1)
                            add_to_dict(new_prod_dict[outer_factor_ind], ordind(factor[1], h[0]), 1)
                            add_to_dict(new_prod_dict[outer_factor_ind], factor, -1)
                    

                            new_coefficient = CumulantPoly(((0, 0),), -outer_factor[factor])
                            new_coefficient * term_to_diff.coeff
                            
                            new_term = Term(term_to_diff.F_derivative, new_prod_dict, term_to_diff.q_deg, term_to_diff.N_deg, new_coefficient)
                            new_h_list = copy.deepcopy(term_to_diff.h_list)
                            new_term.h_list = new_h_list
                            new_term_collection.add_term(new_term)

                            # third contribution, the contribution through the z
                            if use_random_edge:
                                v = term_to_diff.get_fresh_index(0)
                                new_dict = copy.deepcopy(term_to_diff.prod_dict)
                                add_to_dict(new_dict[outer_factor_ind], ordind(factor[0], v), 1)
                                add_to_dict(new_dict[outer_factor_ind], ordind(factor[1], v), 1)
                                add_to_dict(new_dict[outer_factor_ind], factor, -1)

                                new_coefficient = CumulantPoly(((0, 0),), 4 * (outer_factor[factor]))
                                new_coefficient * term_to_diff.coeff
                                
                                new_term = Term(term_to_diff.F_derivative, new_dict, term_to_diff.q_deg, term_to_diff.N_deg, new_coefficient)
                                new_h_list = copy.deepcopy(term_to_diff.h_list)
                                new_term.h_list = new_h_list
                                new_term.add_h(h)
                                new_term_collection.add_term(new_term)
                                # print(new_term)



                # update the term_collection
                curr_term_collection = new_term_collection

                # print("\n\nDerivative", diff_ind,":")
                # curr_term_collection.print_terms_in_order(3)
                # store the latest computed results
                computed_curr_term_collections[diff_ind] = copy.deepcopy(curr_term_collection)
            
            # curr_term_collection now contains the terms for the current cumulant_power differentiation 
            # build the cumulant polynomial and multiply the coefficient with it
            curr_term_collection_copy = copy.deepcopy(curr_term_collection)
            if cumulant_order >= 4:
                cumulant_multiplier = CumulantPoly(((0,0), (cumulant_order, 1)), 1)
            else:
                cumulant_multiplier = CumulantPoly(((0, 0),), 1)
            curr_term_collection_copy.mult_with_constant(cumulant_multiplier)

            for term_key in curr_term_collection_copy.terms:
                curr_term_collection_copy.terms[term_key].q_deg += (cumulant_order - 2)
                result_term_collection.add_term(curr_term_collection_copy.terms[term_key])

        # erase terms with left over h:s, since they will belong to \Tau_F
        result_term_keys = copy.deepcopy(list(result_term_collection.terms.keys()))
        for result_term_key in result_term_keys:
            if len(result_term_collection.terms[result_term_key].h_list) > 0:
                del result_term_collection.terms[result_term_key]

        result_term_collection.rearrange_variables_all_terms()

        return result_term_collection

def produce_almost_unique_identifier(term: Term):

    outer_factor_sizes = []
    for outer_factor in term.prod_dict:
        size = 0
        for factor in outer_factor:
            size += outer_factor[factor]
        outer_factor_sizes.append(size)

    almost_unique_identifier = []
    nbr_vars = term.get_nbr_vars()
    for i in range(nbr_vars):
        var_i_occ = []
        for j in range(nbr_vars):
            outer_factor_occ = []
            for outer_factor_ind, outer_factor in enumerate(term.prod_dict):
                if ordind(i,j) in outer_factor:
                    outer_factor_occ.append((outer_factor[ordind(i,j)], outer_factor_sizes[outer_factor_ind]))
                else:
                    outer_factor_occ.append((0, outer_factor_sizes[outer_factor_ind]))
            var_i_occ.append(tuple(sorted(outer_factor_occ)))
        almost_unique_identifier.append(tuple(sorted(var_i_occ)))
    almost_unique_identifier = tuple(sorted(almost_unique_identifier))

    return almost_unique_identifier

def two_terms_equivalent(term_a: Term, term_b: Term) -> bool:
    """
    returns True if the terms are equivalent up to permutation of indices
    recursively tries to permute the indices of one term into the other
    """
    if term_a.F_derivative != term_b.F_derivative or term_a.get_degree() != term_b.get_degree() or term_a.q_deg != term_b.q_deg or term_a.N_deg != term_b.N_deg or term_a.get_nbr_vars() != term_b.get_nbr_vars():
        return False
    
    outer_factor_lengths_a = sorted([len(outer_factor) for outer_factor in term_a.prod_dict])
    outer_factor_lengths_b = sorted([len(outer_factor) for outer_factor in term_b.prod_dict])
    if outer_factor_lengths_a != outer_factor_lengths_b:
        return False
    
    if produce_almost_unique_identifier(term_a) != produce_almost_unique_identifier(term_b):
        return False

    if term_a.F_derivative == 1:
        # use fast third party library solving the graph isomorphism problem
        edges_a = []
        for edge in term_a.prod_dict[0]:
            for _ in range(term_a.prod_dict[0][edge]):
                edges_a.append(edge)
        edges_b = []
        for edge in term_b.prod_dict[0]:
            for _ in range(term_b.prod_dict[0][edge]):
                edges_b.append(edge)
        G_a = nx.MultiGraph(edges_a)
        G_b = nx.MultiGraph(edges_b)
        if nx.vf2pp_is_isomorphic(G_a, G_b, node_label=None):
            return True
        else:
            return False
                
    def index_occurance_trail(term, index):
        count = []
        for outer_factor in term.prod_dict:
            diag_count = 0
            off_diag_count = 0
            for factor in outer_factor:
                if factor[0] == index and factor[0] == factor[1]: 
                    diag_count += 2
                elif factor[0] == index or factor[1] == index:
                    off_diag_count += 1
            count.append((diag_count, off_diag_count))
        count.sort()
        return count

    def search_permutations(term_a, term_b, current_ind, taken_indices: list[bool], current_pairing):
        if current_ind == term_a.get_nbr_vars():
            # all indices are matched, check if the terms are equivalent
            a_copy = copy.deepcopy(term_a)
            for ind, outer_factor in enumerate(a_copy.prod_dict):
                outer_factor_list = list(outer_factor.items())
                new_outer_factor = [(ordind(current_pairing[item[0][0]], current_pairing[item[0][1]]), item[1]) for item in outer_factor_list]
                a_copy.prod_dict[ind] = dict(new_outer_factor)
            if a_copy.get_key() == term_b.get_key():
                return True
            else:
                return False

        # match current_ind with an index from b.
        # count occurances in term_a
        a_count = index_occurance_trail(term_a, current_ind)
        for b_ind in range(term_b.get_nbr_vars()):
            if taken_indices[b_ind]:
                continue
            b_count = index_occurance_trail(term_b, b_ind)
            if a_count == b_count:
            # this means current_ind -> b_ind is a possible matching and we shall continue the recursion
                current_pairing[current_ind] = b_ind
                taken_indices[b_ind] = True
                if search_permutations(term_a, term_b, current_ind+1, taken_indices, current_pairing):
                    return True
                taken_indices[b_ind] = False
        return False

    
    return search_permutations(term_a, term_b, 0, [False for _ in range(term_a.get_nbr_vars())], [-1 for _ in range(term_a.get_nbr_vars())])

class Ab_term_container:

    def __init__(self):
        self.storage_dict = {}
        self.next_term_ind = 0
        self.term_to_ind = {}
        self.basis_terms = []

    def contains(self, term: Term) -> bool:
        if term.get_key() in self.term_to_ind:
            return True
        almost_uid = produce_almost_unique_identifier(term)
        if almost_uid in self.storage_dict:
            for stored_term in self.storage_dict[almost_uid]:
                if two_terms_equivalent(stored_term, term):
                    return True
        return False
    
    def store_and_get_term_ind(self, term: Term) -> int:
        term_key = term.get_key()
        if term_key in self.term_to_ind:
            return self.term_to_ind[term_key]
        almost_uid = produce_almost_unique_identifier(term)
        if almost_uid in self.storage_dict:
            for stored_term in self.storage_dict[almost_uid]:
                if two_terms_equivalent(stored_term, term):
                    self.term_to_ind[term_key] = self.term_to_ind[stored_term.get_key()] 
                    return self.term_to_ind[term_key]
        # at this point we know that term is not contained in our ab_term_container
        self.term_to_ind[term_key] = self.next_term_ind
        self.next_term_ind += 1
        if almost_uid not in self.storage_dict:
            self.storage_dict[almost_uid] = []
        term_copy = copy.deepcopy(term)
        self.storage_dict[almost_uid].append(term_copy)
        self.basis_terms.append(term_copy)
        return self.term_to_ind[term_key]
        

def generate_term_obj_from_key(term_key):
    q_deg = term_key[0]
    N_deg = term_key[1]
    F_derivative = term_key[2]
    prod_dict = [dict(outer_factor) for outer_factor in term_key[3]]
    coeff = CumulantPoly(((0, 0),), 1.0)
    term_obj = Term(F_derivative, prod_dict, q_deg, N_deg, coeff)
    term_obj.rearrange_vars()
    return term_obj

def generate_main_time_derivative_expansion():
    main_expansion = Term(1, [{(0, 1): 1}], 0, 1, CumulantPoly(((0, 0),(exp_variable_index, 1)), -0.5))
    main_expansion.add_h((0, 1))
    term_collection = main_expansion.expand_hs()
    
    # remove the first term in the cumulant expansion
    term_keys = list(term_collection.terms.keys())
    for term_key in term_keys:
        if term_collection.terms[term_key].q_deg == 0:
            del term_collection.terms[term_key]

    return term_collection

def generate_random_edge_term():
    global use_random_edge
    deterministic_shift_coeff_q2 = CumulantPoly(((4, 1),(exp_variable_index, 1)), 12)
    deterministic_shift_coeff_q4 = CumulantPoly(((4, 2),(exp_variable_index, 1)), -4*9)
    deterministic_shift_coeff_q4 + CumulantPoly(((6, 1),(exp_variable_index, 1)), 120*3)

    edge_term = Term(1, [{(0, 0): 1}], 2, 1, CumulantPoly(((0, 0),), 1.0))
    edge_term.coeff * deterministic_shift_coeff_q2
    edge_term_collection = TermCollection()
    edge_term_collection.add_term(edge_term)

    edge_term_q4 = Term(1, [{(0, 0): 1}], 4, 1, CumulantPoly(((0, 0),), 1.0))
    edge_term_q4.coeff * deterministic_shift_coeff_q4
    edge_term_collection.add_term(edge_term_q4)

    if use_random_edge:
        random_edge_term = Term(1, [{(0, 0): 1}], 0, 1, CumulantPoly(((0, 0),(exp_variable_index,1)), 1.0))
        random_edge_term.add_h((1, 2))
        random_edge_term.add_h((1, 2))
        cumulant_expansion = random_edge_term.expand_hs()

        expanded_term_keys = list(cumulant_expansion.terms.keys())
        for expanded_term in expanded_term_keys:
            if cumulant_expansion.terms[expanded_term].q_deg == 0:
                del cumulant_expansion.terms[expanded_term]
            
        edge_term_collection + cumulant_expansion

    return edge_term_collection

def generate_second_space_term() -> TermCollection:
    term_collection = TermCollection()
    term1 = Term(2, [{(0,0): 1}, {(1,1):1}], 2, 1, CumulantPoly(((4, 1),(exp_variable_index, 1)), 12))
    term2 = Term(1, [{(0,1) : 2}], 2, 1, CumulantPoly(((4, 1),(exp_variable_index, 1)), 12))
    term_collection.add_term(term1)
    term_collection.add_term(term2)
    return term_collection


def list_contains_eq_term(a: list, term: Term):
    for other_term in a:
        if two_terms_equivalent(other_term, term):
            return True
    return False


def save_object(obj: list[TermCollection | Term] | Ab_term_container, save_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, f"{save_name}.pickle"), "wb") as f:
        pickle.dump(obj, f)

def load_object(save_name) -> list[TermCollection | Term] | Ab_term_container:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if f"{save_name}.pickle" in os.listdir(dir_path):
        print("Loading", os.path.join(dir_path, f"{save_name}.pickle"))
        with open(os.path.join(dir_path, f"{save_name}.pickle"), "rb") as f:
            obj = pickle.load(f)
    return obj


def sanity_check_identities(identities: list[TermCollection]):
    for identity in identities:
        nbr_type_0 = 0
        nbr_type_a = 0
        nbr_type_ab = 0
        for term_key in identity.terms:
            term = identity.terms[term_key]
            var_occurances = term.get_nbr_var_occurances()
            nbr_occ_4 = 0
            for var in var_occurances:
                assert var_occurances[var] == 2 or var_occurances[var] == 4, "This term is unmatched"
                if var_occurances[var] == 4:
                    nbr_occ_4 += 1
            if nbr_occ_4 == 2:
                nbr_type_ab += 1
            elif nbr_occ_4 == 1:
                nbr_type_a += 1
            elif nbr_occ_4 == 0:
                nbr_type_0 += 1
            else:
                assert False, "Too many indices"
        if nbr_type_0 > 0 and nbr_type_ab > 0:
            assert False, "An identity like this should not exist."
    print("Identities passed sanity checks!")


def main():

    global permutation_groups_global

    total_degree_limit = 100
    degree_filter = 100
    q_degree_filter = 3 # note, this version can only handle q_degree_filter = 3, i.e. just considering the cancellation of the terms with coefficient kappa_4.
    dir_path = os.path.dirname(os.path.realpath(__file__))

    save_name = "general_F_exp_final"
    identities_save_name = "general_F_identities_final"
    basis_terms_save_name = "general_F_basis_terms_final"
    add_ab_step_precomputed = True
    identities_precomputed = True
    basis_terms_precomputed = True
    big_eq_system_precomputed = True

    # compute the time derivative d/dt E[F(X(t))]
    if not add_ab_step_precomputed:
        term_collection = generate_main_time_derivative_expansion()
        term_collection.print_terms_in_order()
        edge_term_collection = generate_random_edge_term()

        edge_term_collection.filter_and_group_equivalent(total_degree_limit, q_degree_filter, degree_filter)
        edge_term_collection.print_terms_in_order()

        term_collection + edge_term_collection
        term_collection.filter_and_group_equivalent(total_degree_limit, q_degree_filter, degree_filter)
                
        term_collection.filter_and_group_equivalent(total_degree_limit, q_degree_filter, degree_filter)
        term_collection.save_instance(save_name)

    term_collection = load_term_collection(save_name)
    term_collection.filter_and_group_equivalent(total_degree_limit, q_degree_filter, degree_filter)
    print(f"\n\nTerms before manipulating them: ({len(term_collection.terms)} unique terms)\n\n")
    term_collection.print_terms_in_order(print_keys=False)
    # compute the identities as described in the article
    if not identities_precomputed:
        identities_la = generate_all_ab_identities(5, 8, total_degree_limit, q_degree_filter, degree_filter, print_out=False)
        save_object(identities_la, identities_save_name)

    identities_la = load_object(identities_save_name)
    
    sanity_check_identities(identities_la)

    if permutation_groups_global is not None:
        with open(os.path.join(dir_path, "saved_permutation_groups_general.pickle"), "wb") as f:
            pickle.dump(permutation_groups_global, f)
    
    if not basis_terms_precomputed:
        ab_term_container = pick_out_basis_terms(term_collection, identities_la)
        save_object(ab_term_container, basis_terms_save_name)
    
    ab_term_container = load_object(basis_terms_save_name)

    term_collection_vector, span_matrix, span_matrix_frac, term_collection_vector_frac = build_span_matrix_and_save_it(term_collection, identities_la, ab_term_container.basis_terms, ab_term_container.term_to_ind)

    if not big_eq_system_precomputed:
        print("Run gaussian_elim.cpp on the file 'perfect_cancellation_linear_system.in' and store the output in 'big_eq_sol.in', then set big_eq_system_precomputed = True and rerun the code.")
        return

    if big_eq_system_precomputed:
        solve_linear_equation_system_and_check_cancellation(ab_term_container.basis_terms, term_collection_vector, span_matrix, span_matrix_frac, term_collection_vector_frac, save_name_of_precomputed_frac_sol="big_eq_sol.in")
    

def generate_identity_using_off_diag_expansion(start_term, outer_factor_to_expand, factor_to_expand, total_degree_limit, q_degree_filter, degree_filter, switch_indices = False, print_out = False):
    global alp
    term_collection = TermCollection()
    term_collection.add_term(start_term)
    if print_out:
        print(f"Expanding G_{{{alp[factor_to_expand[0]]}{alp[factor_to_expand[1]]}}} in {Term.latex_string_from_outer_factor(outer_factor_to_expand)}")
        print(start_term)
    term_collection.expand_off_diagonal_with_z_rule(start_term.get_key(), outer_factor_to_expand, factor_to_expand, switch_indices)

    start_term.coeff * CumulantPoly(((0, 0),), -1.0)
    term_collection.add_term(start_term)

    term_collection.filter_and_group_equivalent(total_degree_limit, q_degree_filter, degree_filter, use_pre_saved_permutation_groups=False)
    if print_out:
        print("Generated identitiy using expansion rule 1:")
        print("0 = ")
        term_collection.print_terms_in_order(print_keys=False)        
    term_collection.mult_with_constant(CumulantPoly(((0,0),), 2.0))
 
    return term_collection

def generate_identity_using_diag_expansion(start_term, outer_factor_to_expand, total_degree_limit, q_degree_filter, degree_filter, reuse_index = None, print_out = False):
    global alp

    term_collection = TermCollection()
    term_collection.add_term(start_term)

    if reuse_index is None:
        term_collection.add_m_with_z_rule(start_term.get_key(), outer_factor_to_expand)
    else:
        term_collection.add_m_with_z_rule(start_term.get_key(), outer_factor_to_expand, reuse_index=reuse_index)
    if print_out:
        print(f"Expanding {Term.latex_string_from_outer_factor(outer_factor_to_expand)} in")
        print(start_term)

    start_term.coeff * CumulantPoly(((0, 0),), -1.0)
    term_collection.add_term(start_term)
    
    term_collection.filter_and_group_equivalent(total_degree_limit, q_degree_filter, degree_filter, use_pre_saved_permutation_groups=False)
    if print_out:
        if reuse_index is None:
            print("Generated identitiy using expansion rule 2:")
            print("0 = ")
            term_collection.print_terms_in_order(print_keys=False)     
        else:
            print(f"Generated identitiy using expansion rule 3 with reused index {alp[reuse_index]}:")
            print("0 = ")
            term_collection.print_terms_in_order(print_keys=False)  

    term_collection.mult_with_constant(CumulantPoly(((0,0),), 2.0))

    return term_collection

def generate_all_ab_identities(max_nbr_indices, max_F_derivatives, total_degree_limit, q_degree_filter, degree_filter, print_out = False):
    a = 0
    b = 1
    # in the calls to generate_type_ab_terms it is max_nbr_indices - 3 since we will add at least one more index to each of the generated terms. 
    if throw_away_higher_derivatives:
        type_ab_terms = generate_type_ab_terms(max_nbr_indices-3, 1, print_out=print_out)
    else:
        type_ab_terms = generate_type_ab_terms(max_nbr_indices-3, max_F_derivatives, print_out=print_out)
    percentage_finished = 0
    identities_la = []
    for term_ind, term in enumerate(type_ab_terms):
        var_occ = term.get_nbr_var_occurances()
        new_percentage = math.floor(term_ind/len(type_ab_terms)*100)
        if new_percentage > percentage_finished:
            percentage_finished = new_percentage
            print(new_percentage, "% finished computing identities")
        factors_to_expand = []  
        
        has_empty_factor = False
        for outer_factor in term.prod_dict:
            if len(outer_factor) == 0:
                has_empty_factor = True
            for factor in outer_factor:
                if factor[0] != factor[1]:
                    factors_to_expand.append((factor, tuple(outer_factor.items())))                  

        if len(term.get_var_indices()) <= max_nbr_indices-1 and not has_empty_factor:
            for factor_to_expand in factors_to_expand:
                modified_term1 = copy.deepcopy(term) 
                modified_term2 = copy.deepcopy(term)                
                identities_la.append(generate_identity_using_off_diag_expansion(modified_term1, factor_to_expand[1], factor_to_expand[0], total_degree_limit, q_degree_filter, degree_filter, True, print_out=print_out))
                identities_la.append(generate_identity_using_off_diag_expansion(modified_term2, factor_to_expand[1], factor_to_expand[0], total_degree_limit, q_degree_filter, degree_filter, False, print_out=print_out))

        if len(term.get_var_indices()) <= max_nbr_indices-2:
            for outer_factor in term.prod_dict:
                if has_empty_factor and len(outer_factor) > 0:
                    continue
                modified_term1 = copy.deepcopy(term) 
                identities_la.append(generate_identity_using_diag_expansion(modified_term1, tuple(outer_factor.items()), total_degree_limit, q_degree_filter, degree_filter, print_out=print_out))

        if len(term.get_var_indices()) <= max_nbr_indices-1:
            for outer_factor in term.prod_dict:
                if has_empty_factor and len(outer_factor) > 0:
                    continue
                if a not in var_occ or var_occ[a] < 4:
                    modified_term1 = copy.deepcopy(term) 
                    identities_la.append(generate_identity_using_diag_expansion(modified_term1, tuple(outer_factor.items()), total_degree_limit, q_degree_filter, degree_filter, reuse_index = a, print_out=print_out))
                if b not in var_occ or var_occ[b] < 4:
                    modified_term1 = copy.deepcopy(term) 
                    identities_la.append(generate_identity_using_diag_expansion(modified_term1, tuple(outer_factor.items()), total_degree_limit, q_degree_filter, degree_filter, reuse_index = b, print_out=print_out))

    return identities_la



            
def pick_out_basis_terms(term_collection, identities_la):
    print("Picking out basis terms")
    # loop through identities and term_collection to pick out the basis terms, this is regardless of cumulant coefficient
    ab_term_container = Ab_term_container()

    percentage_finished = 0
    for id_ind, identity in enumerate(identities_la):

        # print out progress:
        new_percentage = math.floor(id_ind/len(identities_la)*100)
        if new_percentage > percentage_finished:
            percentage_finished = new_percentage
            print(new_percentage, "% finished finding basis terms")

        for term_key in identity.terms:
            term = identity.terms[term_key]
            ab_term_container.store_and_get_term_ind(term) 


    for term_key in term_collection.terms:
        term = term_collection.terms[term_key]
        ab_term_container.store_and_get_term_ind(term)
    
    return ab_term_container
    
def save_coo_format(m, n, row_inds, col_inds, data, rhs):
    data_to_add = []
    rhs_non_zero = 0
    for i, x in enumerate(rhs):
        if x != 0:
            rhs_non_zero += 1
            data_to_add.append((i, n, int(x), 1))
    for r, c, d in zip(row_inds, col_inds, data):
        data_to_add.append((r, c, int(d), 1))
    data_to_add.sort()
    data_to_add = [f"{x[0]+1} {x[1]+1} {x[2]} {x[3]}" for x in data_to_add]
    file_name = "perfect_cancellation_linear_system.in"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, f"{file_name}"), "w") as f:
        f.write(f"{m} {n} {1} {len(data) + rhs_non_zero}\n")
        f.write("\n".join(data_to_add))
    
def load_frac_sol(file_name):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(dir_path, f"{file_name}"), "r") as f:
       raw_sol = f.read()
    split_sol = raw_sol.split("\n")
    x = np.zeros(len(split_sol), dtype=object)
    for i in range(len(split_sol)):
        x[i] = Fraction(split_sol[i])
    return x

def build_span_matrix_and_save_it(term_collection: TermCollection, precomputed_identities, basis_terms, basis_term_indices):

    identities_la = precomputed_identities

    # the basis terms should be different if they have a different coefficient. And the identities have to be added once for each of the coefficients.
    occuring_cumulants = set()
    for term_key in term_collection.terms:
        term = term_collection.terms[term_key]
        for cumulant_term in term.coeff.poly:
            if term.coeff.poly[cumulant_term] != 0.0:
                occuring_cumulants.add(tuple(sorted(cumulant_term)))

    print("\n\nBasis terms:\n\n")
    new_basis_terms = []
    basis_terms_cumulant_factors = []
    for i, term in enumerate(basis_terms):
        for cumulant_term in occuring_cumulants:
            term_copy = copy.deepcopy(term)
            term_copy.coeff = CumulantPoly(cumulant_term, 1.0)
            new_basis_terms.append(term_copy)
            basis_terms_cumulant_factors.append(cumulant_term)
            print(new_basis_terms[-1], "basis term index", len(new_basis_terms)-1)

    basis_terms = new_basis_terms
    nbr_basis_terms = len(basis_terms)

    # span_matrix = np.zeros((nbr_basis_terms, len(identities_la)), dtype=float)
    span_matrix_frac = np.zeros((nbr_basis_terms, len(identities_la)), dtype=object)
    row_ind = []
    col_ind = []
    data = []

    print("Coefficient matrix is of dimension", nbr_basis_terms, "x", len(identities_la))
    print("Coefficient matrix:")

    percentage_finished = 0
    for j in range(len(identities_la)):
        non_zero_elements = []
        new_percentage = math.floor(j/len(identities_la)*100)
        if new_percentage > percentage_finished:
            percentage_finished = new_percentage
            # print(new_percentage, "% finished building span_matrix")
        for cumulant_term in occuring_cumulants:
            term_collection_copy = copy.deepcopy(identities_la[j])
            term_collection_copy.mult_with_constant(CumulantPoly(cumulant_term, 1.0))
 
            for term_key in term_collection_copy.terms:
                term = term_collection_copy.terms[term_key]
                basis_term_ind = basis_term_indices[term.get_key()]
                row_ind.append(basis_term_ind)
                col_ind.append(j)
                data.append(term.coeff.get_coeff_of_term(basis_terms_cumulant_factors[basis_term_ind]))
                span_matrix_frac[basis_term_ind][j] = Fraction(int(term.coeff.get_coeff_of_term(basis_terms_cumulant_factors[basis_term_ind])))
                non_zero_elements.append((basis_term_ind+1, span_matrix_frac[basis_term_ind][j]))
                
        non_zero_elements = sorted(non_zero_elements, key=lambda x: x[0])
        non_zero_elements = [f"{x[0]}th entry: {str(x[1])}" for x in non_zero_elements]
        print(f"{j+1}th column of coefficient matrix: {', '.join(non_zero_elements)}")

    row_ind = np.array(row_ind)
    col_ind = np.array(col_ind)
    data = np.array(data)
    span_matrix = coo_matrix((data, (row_ind, col_ind)), shape=(nbr_basis_terms, len(identities_la)))

    
    non_zero_elements = []
    term_collection_vector = np.zeros(nbr_basis_terms, dtype = float)
    term_collection_vector_frac = np.zeros(nbr_basis_terms, dtype = object)
    for term_key in term_collection.terms:
        term = term_collection.terms[term_key]
        basis_term_ind = basis_term_indices[term.get_key()]

        term_collection_vector[basis_term_ind] = float(int(term.coeff.get_coeff_of_term(basis_terms_cumulant_factors[basis_term_ind])))
        term_collection_vector_frac[basis_term_ind] = Fraction(int(term.coeff.get_coeff_of_term(basis_terms_cumulant_factors[basis_term_ind])))
        non_zero_elements.append((basis_term_ind+1, term_collection_vector_frac[basis_term_ind]))
    
    # print("Term collection:")
    # term_collection.print_terms_in_order()
    
    print("Rhs of linear equation (denoted (b_l)_{l=1}^{L} in the article):")
    non_zero_elements = sorted(non_zero_elements, key=lambda x: x[0])
    non_zero_elements = [f"{x[0]}th entry: {str(x[1])}" for x in non_zero_elements]
    print(f"d/dt E[F(X(t))] on vector form: {', '.join(non_zero_elements)}")

    save_coo_format(nbr_basis_terms, len(identities_la), row_ind, col_ind, data, term_collection_vector)

    return term_collection_vector, span_matrix, span_matrix_frac, term_collection_vector_frac

def solve_linear_equation_system_and_check_cancellation(basis_terms, term_collection_vector, span_matrix, span_matrix_frac,
                              term_collection_vector_frac, save_name_of_precomputed_frac_sol = None):

    nbr_basis_terms = len(basis_terms)

    # Solve the linear system approximately to get a feeling whether there exists an exact solution or not. 
    span_matrix = csc_matrix(span_matrix)
    x_float, istop, itn, normr = lsqr(span_matrix, term_collection_vector, atol = 0, btol = 0, conlim=0, iter_lim=50000)[:4]
    print(x_float, istop, itn, normr)
    print("Max x-value:", np.max(x_float))
    print("mean x-value:", np.mean(x_float))
    remainder = term_collection_vector - span_matrix @ x_float
    print("Remainder norm:", np.linalg.norm(remainder))
    nbr_non_zero_x = 0
    for xi in x_float:
        if xi > 1e-6:
            nbr_non_zero_x += 1
    print("Nbr of identities used for Krylov subspace method:", nbr_non_zero_x)
    

    if save_name_of_precomputed_frac_sol is not None:
        x_frac = load_frac_sol(save_name_of_precomputed_frac_sol)
    else:
        x_frac, _ = gaussian_elim.solveLinear(span_matrix_frac, term_collection_vector_frac)
    
    print("Exact solution in Fractions:", x_frac)
    print("Largest absolute value in solution:", max(abs(x_frac)))
    Ax = np.zeros(span_matrix_frac.shape[0], dtype=object)
    for j in range(span_matrix_frac.shape[1]):
        if x_frac[j] != Fraction(0, 1):
            Ax += x_frac[j] * span_matrix_frac[:, j]
    remainder = term_collection_vector_frac - Ax

    print("Number of identities used for exact fraction solution:", sum([0 if x == Fraction(0, 1) else 1 for x in x_frac]))
    
    remainder_equal_zero = True
    for i in range(nbr_basis_terms):
        if remainder[i] != Fraction(0, 1):
            remainder_equal_zero = False

    if remainder_equal_zero:
        print("Reduction obtained!")
    else:
        print("Term_collection NOT 0!")
        for i in range(nbr_basis_terms):
            if np.abs(remainder[i]) != Fraction(0, 1):
                print("Basis term", basis_terms[i], basis_terms[i].get_key(), "\nnot reduced, remaining entry:", remainder[i])

    non_zero_elements = []
    for i in range(x_frac.shape[0]):
        if x_frac[i] != 0:
            non_zero_elements.append((i+1, x_frac[i]))
    
    non_zero_elements = sorted(non_zero_elements, key=lambda x: x[0])
    non_zero_elements = [f"{x[0]}th entry: {str(x[1])}" for x in non_zero_elements]
    print(f"Solution to linear equation system: {', '.join(non_zero_elements)}")

    for c in range(span_matrix.shape[1]):
        if x_frac[c] != Fraction(0, 1):
            print("\n\nUsed identity:")
            print(f"\n\n{x_frac[c]} * (")
            for r in range(span_matrix_frac.shape[0]):
                if span_matrix_frac[r][c] == 0:
                    continue
                term_copy = copy.deepcopy(basis_terms[r])
                term_copy.coeff * CumulantPoly(((0,0),), span_matrix_frac[r][c])
                print(term_copy, "basis term index", r)
            print(")")

    if remainder_equal_zero:
        print("Reduction obtained!")

    return

def proper_subsets(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1, 2, 3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(0, len(s)+1))

def generate_type_ab_terms(number_fresh_indices, max_F_derivatives, print_out = False):
    """
    Generates all type-0 terms, type-A terms, and type-AB terms that contain at most number_fresh_indices+2 different indices.
    """

    def distribute_indices_in_ab_factors(permutations, ind, nbr_factors, remaining_indices, determined_indices):
        if ind == nbr_factors-1:
            determined_indices[ind] = remaining_indices
            permutations.append(tuple(determined_indices))
            return
        for i in range(remaining_indices+1):
            determined_indices[ind] = i
            distribute_indices_in_ab_factors(permutations, ind+1, nbr_factors, remaining_indices-i, determined_indices)
        
    #type ab terms:
    # aa aa bb bb * Type_0
    # ab ab aa bb * Type_0
    # ab ab ab ab * Type_0

    # type a terms:
    # aa aa * Type_0

    # type 0 terms:
    # Type_0

    ab_partitions = []
    a_partitions = []
    
    total_nbr_indices = number_fresh_indices
    for i in range(total_nbr_indices+1):
        distribute_indices_in_ab_factors(ab_partitions, 0, 4, i, [-1]*4)
    for i in range(total_nbr_indices+3):
        distribute_indices_in_ab_factors(a_partitions, 0, 2, i, [-1]*2)
    
    if print_out:
        print("ab_partitions:", ab_partitions)
        print("aa_partitions:", a_partitions)
    
    type_0_terms = [()]
    unique_type_0_terms = set()
    number_fresh_indices += 4
    for nbr_factors in range(1, number_fresh_indices+1):
        total_nbr_indices = number_fresh_indices - nbr_factors
        for nbr_indices in range(total_nbr_indices+1):
            distribute_indices_in_ab_factors(type_0_terms, 0, nbr_factors, nbr_indices, [-1]*nbr_factors)
    for i in range(len(type_0_terms)):
        type_0_terms[i] = tuple(sorted((x+1 for x in type_0_terms[i])))
        unique_type_0_terms.add(type_0_terms[i])
    number_fresh_indices -= 4

    unique_type_0_terms = sorted(list(unique_type_0_terms))

    if print_out:
        print("type_0_terms:", unique_type_0_terms)

    
    def add_loop_to_dict(dic, start, end, free_ind, loop_len):
        curr_len = 1
        curr_ind = start
        while curr_len < loop_len:
            add_to_dict(dic, ordind(curr_ind, free_ind), 1)
            curr_ind = free_ind
            free_ind += 1
            curr_len += 1
        add_to_dict(dic, ordind(curr_ind, end), 1)
        return free_ind

    a = 0
    b = 1
    next_free_ind = 2
    type_ab_terms = []
    
    for ab_partition in ab_partitions:
        # aa aa bb bb * Type_0
        new_prod_dict = {}
        next_free_ind = 2
        next_free_ind = add_loop_to_dict(new_prod_dict, a, a, next_free_ind, ab_partition[0] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, a, next_free_ind, ab_partition[1] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, b, b, next_free_ind, ab_partition[2] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, b, b, next_free_ind, ab_partition[3] + 1)
        new_term = Term(1, [new_prod_dict], 2, 1, CumulantPoly(((0,0),), 1.0))
        if not list_contains_eq_term(type_ab_terms, new_term):
            type_ab_terms.append(new_term)

        # ab ab aa bb * Type_0
        new_prod_dict = {}
        next_free_ind = 2
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[0] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[1] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, a, next_free_ind, ab_partition[2] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, b, b, next_free_ind, ab_partition[3] + 1)
        new_term = Term(1, [new_prod_dict], 2, 1, CumulantPoly(((0,0),), 1.0))
        if not list_contains_eq_term(type_ab_terms, new_term):
            type_ab_terms.append(new_term)
        
        # ab ab ab ab * Type_0
        new_prod_dict = {}
        next_free_ind = 2
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[0] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[1] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[2] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[3] + 1)
        new_term = Term(1, [new_prod_dict], 2, 1, CumulantPoly(((0,0),), 1.0))
        if not list_contains_eq_term(type_ab_terms, new_term):
            type_ab_terms.append(new_term)

    type_a_terms = []
    for a_partition in a_partitions:
        # aa aa bb bb * Type_0
        new_prod_dict = {}
        next_free_ind = 1
        next_free_ind = add_loop_to_dict(new_prod_dict, a, a, next_free_ind, a_partition[0] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, a, next_free_ind, a_partition[1] + 1)
        new_term = Term(1, [new_prod_dict], 2, 1, CumulantPoly(((0,0),), 1.0))
        if not list_contains_eq_term(type_a_terms, new_term):
            type_a_terms.append(new_term)

    if print_out:
        print("\n\nab_terms:\n\n")
        for term in type_ab_terms:
            print(term)
            print(term.get_key())
        print("\n\na_terms:\n\n")
        for term in type_a_terms:
            print(term)
            print(term.get_key())
    
    type_0_terms = [Term(1, [{}], 2, 1, CumulantPoly(((0,0),), 1.0))]
    max_nbr_indices = number_fresh_indices + 2
    completed_terms = []
    for term_ind in range(len(type_ab_terms)):
        nbr_used_indices = type_ab_terms[term_ind].get_nbr_vars()
        for unique_type_0_term in unique_type_0_terms:
            nbr_type_0_indices = sum(unique_type_0_term)
            if nbr_type_0_indices <= max_nbr_indices - nbr_used_indices:
                new_term = copy.deepcopy(type_ab_terms[term_ind])
                for loop_size in unique_type_0_term:
                    next_free_ind = new_term.get_fresh_index(0)
                    next_free_ind = add_loop_to_dict(new_term.prod_dict[0], next_free_ind, next_free_ind, next_free_ind+1, loop_size)
                # assert not list_contains_eq_term(completed_terms, new_term)
                completed_terms.append(new_term)

    for term_ind in range(len(type_a_terms)):
        nbr_used_indices = type_a_terms[term_ind].get_nbr_vars()
        for unique_type_0_term in unique_type_0_terms:
            nbr_type_0_indices = sum(unique_type_0_term)
            if nbr_type_0_indices <= max_nbr_indices - nbr_used_indices:
                new_term = copy.deepcopy(type_a_terms[term_ind])
                for loop_size in unique_type_0_term:
                    next_free_ind = new_term.get_fresh_index(0)
                    next_free_ind = add_loop_to_dict(new_term.prod_dict[0], next_free_ind, next_free_ind, next_free_ind+1, loop_size)
                # assert not list_contains_eq_term(completed_terms, new_term)
                completed_terms.append(new_term)

  
    nbr_used_indices = 0
    for unique_type_0_term in unique_type_0_terms:
        nbr_type_0_indices = sum(unique_type_0_term)
        if nbr_type_0_indices <= max_nbr_indices - nbr_used_indices:
            new_term = copy.deepcopy(type_0_terms[0])
            for loop_size in unique_type_0_term:
                next_free_ind = new_term.get_fresh_index(0)
                next_free_ind = add_loop_to_dict(new_term.prod_dict[0], next_free_ind, next_free_ind, next_free_ind+1, loop_size)
            # assert not list_contains_eq_term(completed_terms, new_term)
            completed_terms.append(new_term)
    
    if print_out:
        print(f"\n\Pre-split completed terms: ({len(completed_terms)} terms unique terms)\n\n")
        for term in completed_terms:
            print(term)
            print(term.get_key())
    
    partitioned_terms = []
    for term_ind in range(len(completed_terms)):
        term = completed_terms[term_ind]
        nbr_Gs = term.get_nbr_Gs()
        Gs_list = term.get_G_list()
        if nbr_Gs == 0:
            partitioned_terms.append(Term(1, [{}], 2, 1, CumulantPoly(((0,0),), 1.0)))
            continue
        partitions = mit.set_partitions(range(nbr_Gs))
        new_terms = []
        for partition in partitions:
            if (len(partition) > max_F_derivatives):
                continue
            new_term_prod_dict = [dict() for _ in range(len(partition))]
            for part_ind, part in enumerate(partition):
                for i in part:
                    add_to_dict(new_term_prod_dict[part_ind], Gs_list[i], 1)
            new_term = copy.deepcopy(term)
            new_term.prod_dict = new_term_prod_dict
            new_term.F_derivative = len(partition)
            if not list_contains_eq_term(new_terms, new_term):
                new_terms.append(new_term)
                if new_term.F_derivative <= max_F_derivatives-1:
                    new_term_copy = copy.deepcopy(new_term)
                    new_term_copy.F_derivative += 1
                    new_term_copy.prod_dict.append(dict())
                    new_terms.append(new_term_copy)
        partitioned_terms.extend(new_terms)

    if print_out:
        print(f"\n\Partitioned terms: ({len(partitioned_terms)} terms unique terms)\n\n")
        for term in partitioned_terms:
            print(term)
            print(term.get_key())

    return partitioned_terms

def generate_terms_test():
    generate_type_ab_terms(2, 8, print_out=True)

if __name__ == "__main__":
    main()
    # generate_terms_test()

