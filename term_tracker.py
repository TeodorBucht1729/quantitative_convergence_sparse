from fractions import Fraction
from itertools import product, permutations
import os
import pickle
from typing import Self
import copy
import networkx as nx
import numpy as np
import gaussian_elim

h_index_offset = -1000
exp_variable_index = 1001
random_edge_index = 1002

global alp
global latex_print_outs

alp = ['a', 'b', 'v', 'j', 'e', 'f', 'g', 'h']
latex_print_outs = True

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

    def __add__(self, other_poly_obj: Self):
        other_poly = copy.deepcopy(other_poly_obj) # probably changes nothing
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
    
    def __eq__(self, other_term):
        self_copy = copy.deepcopy(self)
        other_copy = copy.deepcopy(other_term)
        other_copy * CumulantPoly(((0, 0),), -1.0)
        self_copy + other_copy        
        if self_copy.is_zero():
            return True
        else:
            return False

    def is_constant(self):
        if len(self.poly.keys()) == 1:
            return True 

    def __mul__(self, other_poly_obj: Self):
        other_poly = copy.deepcopy(other_poly_obj) # probably changes nothing
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
                cumulant_string = "\\kappa_4 e^{-t}(1 - e^{-t})"
            else:
                cumulant_string = ''.join([f"\kappa_{ind}^{p}" if ind != 0 else '1' for (ind, p) in term])
            terms.append(f"{Fraction(self.poly[term])} {cumulant_string}")
        terms.sort()
        return " + ".join(terms)
    
    def reduce_random_edge_contribution(self, reduce_fully):

        new_coeff = CumulantPoly(((0,0),), 0.0)
        for term in self.poly:
            term_list = list(term)
            term_list_copy = copy.deepcopy(term_list)
            for i, factor in enumerate(term_list_copy):
                if factor[0] == random_edge_index:
                    if reduce_fully:
                        term_list.remove(factor)
                    else:
                        term_list[i] = (random_edge_index, 1)
            new_term = tuple(term_list)
            new_coeff + CumulantPoly(new_term, self.poly[term])
        self.poly = new_coeff.poly




def two_terms_equivalent(term_a, term_b) -> bool:
    """
    returns True if the terms are equivalent up to permutation of indices
    """
    assert term_a.h_list == [] and term_b.h_list == [], "Only works for terms without h:s"
    if term_a.get_degree() != term_b.get_degree() or term_a.q_deg != term_b.q_deg or term_a.N_deg != term_b.N_deg:
        return False
    
    edges_a = []
    for edge in term_a.prod_dict:
        for _ in range(term_a.prod_dict[edge]):
            edges_a.append(edge)
    edges_b = []
    for edge in term_b.prod_dict:
        for _ in range(term_b.prod_dict[edge]):
            edges_b.append(edge)
    G_a = nx.MultiGraph(edges_a)
    G_b = nx.MultiGraph(edges_b)
    if nx.vf2pp_is_isomorphic(G_a, G_b, node_label=None):
        # print("Equivalent terms:", term_a, term_b)
        return True
    else:
        return False
    


class TermCollection:
    """
    Represents a sum of standard terms of the form N^(N_deg)/(N^(#I) q^(q_deg)) sum_(I) cumulant_poly E[prod G_(x_iy_i)].
    The terms are saved in the dict self.terms, the keys are of the form (q_deg, N_deg, index_graph),
    an example key is (2, 1, (((0, 2), 2), ((1, 1), 2), ((2, 2), 1))), which represents the term
    N/q^2 sum_(v,a,b) E[G_vb^2 G_aa^2 G_bb]. The values in self.terms are Term-objects.
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
        
    def __eq__(self, other_term_collection):
        copy_self = copy.deepcopy(self)
        copy_other = copy.deepcopy(other_term_collection)
        copy_other.mult_with_constant(CumulantPoly(((0,0),), -1.0))
        copy_self + copy_other
        copy_self.group_equivalent()
        return copy_self.is_zero()

    def is_zero(self):
        for term_key in self.terms:
            if not self.terms[term_key].coeff.is_zero():
                return False
        return True

    def filter_trivial_zero_terms(self):
        term_keys = copy.deepcopy(list(self.terms.keys()))
        for term_key in term_keys:
            if self.terms[term_key].coeff.is_zero() or len(self.terms[term_key].prod_dict) == 0:
                del self.terms[term_key]
        

    def filter_high_degree_terms(self, degree_limit):
        """
        deletes all terms for which degree + q_deg >= degree_limit
        """
        term_keys = copy.deepcopy(tuple(self.terms.keys()))
        for term_key in term_keys:
            if self.terms[term_key].get_degree() + self.terms[term_key].q_deg >= degree_limit:
                del self.terms[term_key]

    def filter_degree_terms(self, degree_limit):
        """
        deletes all terms for which degree >= degree_limit
        """
        term_keys = copy.deepcopy(tuple(self.terms.keys()))
        for term_key in term_keys:
            if self.terms[term_key].get_degree() >= degree_limit:
                del self.terms[term_key]

    def filter_high_q_degree_terms(self, q_limit):
        term_keys = copy.deepcopy(tuple(self.terms.keys()))
        for term_key in term_keys:
            if self.terms[term_key].q_deg >= q_limit:
                del self.terms[term_key]

    def mult_with_constant(self, constant: CumulantPoly):
        for term_key in self.terms:
            self.terms[term_key].coeff * constant

    def save_instance(self, save_name):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, f"{save_name}.pickle"), "wb") as f:
            pickle.dump(self, f)

    def reduce_random_edge_contributions(self, reduce_fully = True):
        for term_key in self.terms:
            self.terms[term_key].coeff.reduce_random_edge_contribution(reduce_fully)

    def filter_term_collection(self, total_degree_filter, q_degree_filter, degree_filter, reduce_fully = True):
        # delete terms that have been cancelled
        term_keys = copy.deepcopy(tuple(self.terms.keys()))
        for term_key in term_keys:
            if self.terms[term_key].coeff.is_zero():
                del self.terms[term_key]
        self.reduce_random_edge_contributions(reduce_fully)
        self.filter_high_degree_terms(total_degree_filter)
        self.filter_high_q_degree_terms(q_degree_filter)
        self.filter_degree_terms(degree_filter)

    def filter_and_group_equivalent(self, total_degree_filter, q_degree_filter, degree_filter, reduce_fully = True):
        # delete terms that have been cancelled
        term_keys = copy.deepcopy(tuple(self.terms.keys()))
        for term_key in term_keys:
            if self.terms[term_key].coeff.is_zero():
                del self.terms[term_key]
        self.reduce_random_edge_contributions(reduce_fully)
        self.filter_high_degree_terms(total_degree_filter)
        self.filter_high_q_degree_terms(q_degree_filter)
        self.filter_degree_terms(degree_filter)
        self.filter_trivial_zero_terms()
        self.group_equivalent()

    def group_equivalent(self, pause = False):
        """ 
        Join terms that are equivalent up to permutation of indices
        """

        check_term_collection_for_strange_terms(self, 0, 1)

        save_name = "saved_permutation_groups_basic_format.pickle"
        term_keys = copy.deepcopy(tuple(self.terms.keys()))
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if save_name in os.listdir(dir_path):
            with open(os.path.join(dir_path, save_name), "rb") as f:
                permutation_groups = pickle.load(f)
        else:
            permutation_groups = {}

        total_coeff_start = CumulantPoly(((0,0),), 0.0)
        for term_key in self.terms:
            total_coeff_start + self.terms[term_key].coeff

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
                    self.terms[term].prod_dict = dict(group_representative[2])
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
                    found_equivalent_terms.append(other_term)
                    if other_term in permutation_groups:
                        existing_group_found = True
                        existing_group_representative = permutation_groups[other_term]
            
            # remove duplicates
            found_equivalent_terms = set(found_equivalent_terms)

            # add all equivalent terms to the same Term-object
            if existing_group_found:
                for term_key in found_equivalent_terms:
                    permutation_groups[term_key] = existing_group_representative
                    if term_key == existing_group_representative:
                        continue
                    # add term_key to existing_group_representative
                    self.terms[term_key].prod_dict = dict(existing_group_representative[2])
                    if existing_group_representative not in self.terms:
                        self.terms[existing_group_representative] = Term(self.terms[term_key].prod_dict, self.terms[term_key].q_deg, self.terms[term_key].N_deg, CumulantPoly(((0,0),), 0))
                    self.terms[existing_group_representative] + self.terms[term_key]
                    del self.terms[term_key]
            else:
                for term_key in found_equivalent_terms:
                    permutation_groups[term_key] = term
                    if term_key == term:
                        continue
                    # add term_key to term
                    self.terms[term_key].prod_dict = dict(term[2])
                    self.terms[term] + self.terms[term_key]
                    del self.terms[term_key]

            if term in self.terms and self.terms[term].coeff.is_zero():
                del self.terms[term]
        
        if pause:
            pass

        total_coeff_end = CumulantPoly(((0,0),), 0.0)
        for term_key in self.terms:
            total_coeff_end + self.terms[term_key].coeff

        assert total_coeff_start == total_coeff_end, "Something wrong when grouping equivalent terms"

        # save permutation groups which may have been updated
        with open(os.path.join(dir_path, save_name), "wb") as f:
            pickle.dump(permutation_groups, f)


    def expand_off_diagonal_with_z_rule(self, term_to_expand, factor_to_expand, switch_indices = False):
        """
        Expand a off-diagonal entry using expansion rule 1 from the article
        If switch_indices is True, this corresponds to Rule 1' from the article
        """
        term = self.terms[term_to_expand]

        j = term.get_fresh_index(0)
        v = factor_to_expand[0]
        a = factor_to_expand[1]
        if switch_indices:
            a, v = v, a

        global alp
        # print(f"Expanding $G_{{{alp[v]}{alp[a]}}}$ in \\begin{{equation*}}", term.latex_str(), f", \end{{equation*}} \nto obtain ")

        new_dict = copy.deepcopy(term.prod_dict)

        # prepare to add h_s 
        add_to_dict(new_dict, ordind(v, a), -1)
        add_to_dict(new_dict, ordind(v, j), 1)

        new_coefficient = CumulantPoly(((0, 0),), 0.5)
        new_coefficient * term.coeff

        new_term = Term(new_dict, term.q_deg, term.N_deg+1, new_coefficient)
        new_term.add_h(ordind(a, j))
        # print("\\begin{equation*}", new_term.latex_str(), ". \end{equation*}")
        new_term.rearrange_vars()
        new_term.N_deg -= 1

        higher_order_contribution_term = copy.deepcopy(term)
        higher_order_contribution_term.q_deg += 2
        higher_order_contribution_term.coeff * CumulantPoly(((0, 0),(201, 1)), -1.0)
        higher_order_contribution_term.rearrange_vars()
        self.add_term(higher_order_contribution_term)

        cumulant_expansion = new_term.expand_hs()   


        del self.terms[term_to_expand] # remove the expanded term

        self + cumulant_expansion
        
    def add_m_with_z_rule(self, term_to_expand, reuse_index = None, print_out = False):
        """
        Expand a 1 using expansion rule 2 or 3 from the article. If reuse_index is specified, Rule 3 is used with that index.
        """
        term = self.terms[term_to_expand]

        if reuse_index is None:
            v = term.get_fresh_index(0)
        else:
            v = reuse_index
        j = term.get_fresh_index(v+1)

        assert v != j, "This should not happen"

        global alp
        # print(f"Expanding $G_{{{alp[v]}{alp[a]}}}$ in \\begin{{equation*}}", str(term), f", \end{{equation*}} \nto obtain ")
        if print_out:
            print(f"Adding {alp[v]} to", term)


        new_prod_dict_1 = copy.deepcopy(term.prod_dict)
        
        add_to_dict(new_prod_dict_1, ordind(v, v), 1)

        new_coefficient_1 = CumulantPoly(((0, 0),), -2.0)
        new_coefficient_1 * term.coeff

        new_term_1 = Term(new_prod_dict_1, term.q_deg, term.N_deg, new_coefficient_1)
        new_term_1.rearrange_vars()
        self.add_term(new_term_1)

            
        higher_order_contribution_term = copy.deepcopy(new_term_1)
        higher_order_contribution_term.q_deg += 2
        higher_order_contribution_term.coeff * CumulantPoly(((0, 0),(201, 1)), 1.0)
        self.add_term(higher_order_contribution_term)

        new_prod_dict_2 = copy.deepcopy(term.prod_dict)
        add_to_dict(new_prod_dict_2, ordind(v, j), 1)

        new_coefficient_2 = CumulantPoly(((0, 0),), 1.0)
        new_coefficient_2 * term.coeff    

        new_term_2 = Term(new_prod_dict_2, term.q_deg, term.N_deg+1, new_coefficient_2)
        new_term_2.add_h(ordind(j, v))
        new_term_2.rearrange_vars()
        # print("\\begin{equation*}", str(new_term), ". \end{equation*}")
        
        new_term_2.N_deg -= 1
        cumulant_expansion = new_term_2.expand_hs()

        if print_out:
            print("to obtain:")
            print(new_term_1)
            cumulant_expansion.print_collection()

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
    
    def print_collection(self, latex_format = False):
        global alp
        all_keys = list(self.terms.keys())
        all_keys = [(k[0], self.terms[k].get_degree(), k[1], k[2], k[3]) for k in all_keys]
        all_keys.sort()
        if latex_format:
            print("Leading terms:\n \\begin{equation*} \n \\begin{split} & \quad")
            term_strings = []

        for ext_key in all_keys:            
            key = (ext_key[0], ext_key[2], ext_key[3], ext_key[4])
            term = self.terms[key]
            if term.coeff.is_zero():
                continue
            
            if not latex_format:
                G_strings = [f"G_{{{alp[f[0][0]]}{alp[f[0][1]]}}}^{f[1]}" if f[1] != 1 else f"G_{{{alp[f[0][0]]}{alp[f[0][1]]}}}" for f in key[2]]
                h_strings = [f"h_{alp[h[0]]}{alp[h[1]]}" for h in term.h_list]
                print(f"N^{term.N_deg}/q^{term.q_deg} ({str(term.coeff)}) E[{''.join(h_strings)}{''.join(G_strings)}], degree {ext_key[1]}")
                print("Key:", key)
            if latex_format:
                coeff_string = f"{str(term.coeff)} \\frac{{N^{term.N_deg}}}{{q^{term.q_deg}N^{term.get_nbr_vars()}}} \sum_{{{','.join([alp[var] for var in term.get_var_indices()])}}} \EX \left["
                G_strings = [f"G_{{{alp[f[0][0]]}{alp[f[0][1]]}}}^{f[1]}" if f[1] != 1 else f"G_{{{alp[f[0][0]]}{alp[f[0][1]]}}}" for f in key[2]]
                h_strings = [f"h_{alp[h[0]]}{alp[h[1]]}" for h in term.h_list]
                term_strings.append(f"{coeff_string}{''.join(h_strings)}{''.join(G_strings)} \\right]")

        if latex_format:
            print(" \\\\ \n& +".join(term_strings))
            print(". \end{split} \n\end{equation*}")



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
    Represents are term of the form N^(N_deg)/(N^(#I) q^(q_deg)) sum_(I) cumulant_poly E[prod G_(x_iy_i)].
    """

    def __init__(self, prod_dict, q_deg, N_deg, coeff: CumulantPoly):

        self.prod_dict = prod_dict
        self.q_deg = q_deg
        self.N_deg = N_deg 
        self.coeff = coeff
        self.h_list = [] # this is a list of tuples where each tuple represent the index
        self.q_cutoff = 3 # this is when to cut off the q:s in the cumulant expansions

    def __add__(self, other_term):
        assert self.get_key() == other_term.get_key(), "terms can't be added"
        assert self.h_list == other_term.h_list, "Not the same number of h:s"
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
        
        key = self.get_key()   
        if not latex_print_outs:
            G_strings = [f"G_{alp[f[0][0]]}{alp[f[0][1]]}^{f[1]}" for f in key[2]]
            h_strings = [f"h_{alp[h[0]]}{alp[h[1]]}" for h in self.h_list]
            return f"N^{self.N_deg}/q^{self.q_deg} {str(self.coeff)} E[{''.join(h_strings)}{''.join(G_strings)}], degree {self.get_degree()}"
        else:
            G_strings = []
            if len(key[2]) > 0:
                G_strings.append("".join([f"G_{{{alp[f[0][0]]}{alp[f[0][1]]}}}{f'^{{{f[1]}}}' if f[1] != 1 else ''}" for f in key[2]]))
            else:
                G_strings.append("1")
            h_strings = [f"h_{{{alp[h[0]]}{alp[h[1]]}}}" for h in self.h_list]
            nbr_vars = self.get_nbr_vars()
            if self.N_deg == 1:
                return f"{str(self.coeff)} \\frac{{N}}{{q^{self.q_deg}N^{nbr_vars}}}  \sum_{{{','.join(alp[:nbr_vars])}}} \EX \left[{''.join(h_strings)}{''.join(G_strings)}\\right]"
            else:
                return f"{str(self.coeff)} \\frac{{N^{self.N_deg}}}{{q^{self.q_deg}N^{nbr_vars}}} \sum_{{{','.join(alp[:nbr_vars])}}} \EX \left[F^{{({self.F_derivative})}}(X(t)){''.join(h_strings)}{''.join(G_strings)}\\right]"
        
    def latex_str(self):
        alp = ['v', 'a', 'b', 'c', 'j', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'o', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'y', 'z']
        coeff_string = f"\\frac{{{str(self.coeff)}N^{self.N_deg}}}{{q^{self.q_deg}N^{self.get_nbr_vars()}}} \sum_{{{','.join([alp[var] for var in self.get_var_indices()])}}} \EX \left["
        G_strings = [f"G_{{{alp[f[0][0]]}{alp[f[0][1]]}}}^{f[1]}" if f[1] != 1 else f"G_{{{alp[f[0][0]]}{alp[f[0][1]]}}}" for f in self.prod_dict.items()]
        h_strings = [f"h_{{{alp[h[0]]}{alp[h[1]]}}}" for h in self.h_list]
        return f"{coeff_string}{''.join(h_strings)}{''.join(G_strings)} \\right]"

    def get_fresh_index(self, start_index):
        ind = start_index
        used_variables = self.get_var_indices()
        while(ind in used_variables):
            ind += 1
        return ind

    def get_var_indices(self):
        vars = set()
        for factor in self.prod_dict:
            vars.add(factor[0])
            vars.add(factor[1])
        for h in self.h_list:
            vars.add(h[0])
            vars.add(h[1])
        return list(vars)
    
    def get_nbr_vars(self):
        return len(self.get_var_indices())

    def get_var_occurances_and_indices(self):
        vars = set()
        var_occ = {}
        for h in self.h_list:
            vars.add(h[0])
            vars.add(h[1])
        for factor in self.prod_dict:
            vars.add(factor[0])
            vars.add(factor[1])
            add_to_dict(var_occ, factor[0], self.prod_dict[factor])
            add_to_dict(var_occ, factor[1], self.prod_dict[factor])
        return var_occ, vars

    def get_m_degree(self):
        ans = 0
        for factor in self.prod_dict:
            if factor[0] == factor[1]:
                ans += 1
        return ans


    def rearrange_vars(self):
        """
        Renames the variables to be 0, 1, ..., nbr_vars-1
        """
        used_vars = self.get_var_indices()
        used_vars.sort()
        nbr_vars = len(used_vars)
        if nbr_vars == 0:
            return
        if used_vars[-1] > len(used_vars)-1:
            # variables should be rearranged
            map_to_new_vars = {}
            for i in range(nbr_vars):
                map_to_new_vars[used_vars[i]] = i
            new_prod_dict = {}
            for factor in self.prod_dict:
                new_prod_dict[ordind(map_to_new_vars[factor[0]], map_to_new_vars[factor[1]])] = self.prod_dict[factor]
            for i in range(len(self.h_list)):
                self.h_list[i] = ordind(map_to_new_vars[self.h_list[i][0]], map_to_new_vars[self.h_list[i][1]])
            self.prod_dict = new_prod_dict


    def get_key(self):
        key = (self.q_deg, self.N_deg, tuple(sorted(tuple(self.prod_dict.items()))), tuple(sorted(self.h_list)))
        return key
    
    def get_degree(self):
        ans = 0
        for factor in self.prod_dict:
            if factor[0] != factor[1]:
                ans += self.prod_dict[factor]
        return ans

    def add_h(self, h_ind):
        self.h_list.append(h_ind)

    def expand_main_time_derivative_terms(self) -> TermCollection:
        """
        Computes the terms in the expansion of the time derivative of self coming from the change in h(t),
        i.e. if we had a fixed edge, this would be the whole expansion. 

        The function essentially performs one differentiation w.r.t. h_ab and then calls expand_hs.
        """

        a = self.get_fresh_index(0)
        b = self.get_fresh_index(a+1)
        # v = self.get_fresh_index(b+1)

        one_diff_term_collection = TermCollection()

        for factor in self.prod_dict:
            # first contribution
            new_dict = copy.deepcopy(self.prod_dict)
            add_to_dict(new_dict, ordind(factor[0], a), 1)
            add_to_dict(new_dict, ordind(factor[1], b), 1)
            add_to_dict(new_dict, factor, -1)
    

            new_coefficient = CumulantPoly(((0, 0),(exp_variable_index, 1)), -self.prod_dict[factor] * 0.25) # the factor 0.25 comes from that we only differentiate with half of the h_s and from the time derivative of e^(-t/2)
            new_coefficient * self.coeff
            
            new_term = Term(new_dict, self.q_deg, self.N_deg + 1, new_coefficient)
            one_diff_term_collection.add_term(new_term)

            # second contribution
            new_dict = copy.deepcopy(self.prod_dict)
            add_to_dict(new_dict, ordind(factor[0], b), 1)
            add_to_dict(new_dict, ordind(factor[1], a), 1)
            add_to_dict(new_dict, factor, -1)
    
            new_coefficient = CumulantPoly(((0, 0),(exp_variable_index, 1)), -self.prod_dict[factor] * 0.25)
            new_coefficient * self.coeff
            
            new_term = Term(new_dict, self.q_deg, self.N_deg + 1, new_coefficient)
            one_diff_term_collection.add_term(new_term)
        
        resulting_term_collection = TermCollection()

        for term_key in one_diff_term_collection.terms:
            term = one_diff_term_collection.terms[term_key]
            term.add_h(ordind(a, b))
            cumulant_expansion = term.expand_hs()

            # remove the first order cancellation
            expanded_term_keys = list(cumulant_expansion.terms.keys())
            for expanded_term in expanded_term_keys:
                if cumulant_expansion.terms[expanded_term].q_deg == self.q_deg:
                    del cumulant_expansion.terms[expanded_term]
            
            resulting_term_collection + cumulant_expansion

        return resulting_term_collection


    def get_edge_term(self) -> TermCollection:
        """
        Computes the terms in the expansion of the time derivative of self coming from the change in z,
        i.e. what is called the edge terms. 
        """
        a = self.get_fresh_index(0)
        i = self.get_fresh_index(a+1)
        j = self.get_fresh_index(i+1)

        resulting_term_collection = TermCollection()

        # loop through according to product rule for derivation
        for factor in self.prod_dict:
            
            # contribution from deterministic edge
            new_dict = copy.deepcopy(self.prod_dict)
            add_to_dict(new_dict, ordind(factor[0], a), 1)
            add_to_dict(new_dict, ordind(factor[1], a), 1)
            add_to_dict(new_dict, factor, -1)
    
            new_coefficient = CumulantPoly(((0, 0),(4, 1),(exp_variable_index,1)), self.prod_dict[factor] * 12)
            new_coefficient * self.coeff

            new_term = Term(new_dict, self.q_deg+2, self.N_deg+1, new_coefficient)
            new_term.rearrange_vars()

            higher_order_contribution = copy.deepcopy(new_term)
            higher_order_contribution.q_deg += 2
            higher_order_contribution.coeff = CumulantPoly(((0, 0),(6, 1),(exp_variable_index,1)), self.prod_dict[factor] * 360)
            higher_order_contribution.coeff + CumulantPoly(((0, 0),(4, 2),(exp_variable_index,1)), self.prod_dict[factor] * (-1.0) * (81 - 72)*4) 
            # higher_order_contribution.coeff + CumulantPoly(((0, 0),(4, 2),(exp_variable_index,1)), self.prod_dict[factor] * (-1.0) * (81)*4)
            higher_order_contribution.coeff * self.coeff

            resulting_term_collection.add_term(new_term)
            resulting_term_collection.add_term(higher_order_contribution)

            # contribution from random edge
            new_dict = copy.deepcopy(self.prod_dict)
            add_to_dict(new_dict, ordind(factor[0], a), 1)
            add_to_dict(new_dict, ordind(factor[1], a), 1)
            add_to_dict(new_dict, factor, -1)
    
            new_coefficient = CumulantPoly(((0, 0),(exp_variable_index,1),(random_edge_index,1)), self.prod_dict[factor])
            new_coefficient * self.coeff

            new_term = Term(new_dict, self.q_deg, self.N_deg+1, new_coefficient)
            new_term.add_h((ordind(i,j)))
            new_term.add_h((ordind(i,j)))
            new_term.rearrange_vars()
            cumulant_expansion = new_term.expand_hs()

            # remove the first order cancellation
            expanded_term_keys = list(cumulant_expansion.terms.keys())
            for expanded_term in expanded_term_keys:
                if cumulant_expansion.terms[expanded_term].q_deg == self.q_deg:
                    del cumulant_expansion.terms[expanded_term]
            
            resulting_term_collection + cumulant_expansion
        
        return resulting_term_collection

    def expand_hs(self, h_to_expand = None, throw_away_terms_with_remaining_hs = True) -> TermCollection:
        """ 
        Perform cumulant expansion with one h.
        """
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

        cumulant_orders = list(range(1, self.q_cutoff + 1 - self.q_deg + 1))
        # print("Cumulant powers:", cumulant_orders)
        result_term_collection = TermCollection()

        computed_curr_term_collections = {}

        for cumulant_order in cumulant_orders:
            # when this cycle of the for-loop is completed, the term with the cumulant_order-cumulant will have been computed
            curr_term_collection = TermCollection()
            curr_term_collection.add_term(self)

            # filter out odd unmatched terms by ignoring the terms with odd cumulants
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

                    # the derivative hits a h
                    h_count = term_to_diff.h_list.count(h)
                    if h_count >= 1:
                        new_dict = copy.deepcopy(term_to_diff.prod_dict)
                        new_coefficient = CumulantPoly(((0, 0),), h_count)
                        new_coefficient * term_to_diff.coeff
                        
                        new_term = Term(new_dict, term_to_diff.q_deg, term_to_diff.N_deg, new_coefficient)
                        new_h_list = copy.deepcopy(term_to_diff.h_list)
                        new_h_list.remove(h)
                        new_term.h_list = new_h_list
                        new_term_collection.add_term(new_term)

                    for factor in term_to_diff.prod_dict:

                        # first contribution
                        new_dict = copy.deepcopy(term_to_diff.prod_dict)
                        add_to_dict(new_dict, ordind(factor[0], h[0]), 1)
                        add_to_dict(new_dict, ordind(factor[1], h[1]), 1)
                        add_to_dict(new_dict, factor, -1)
                

                        new_coefficient = CumulantPoly(((0, 0),), -term_to_diff.prod_dict[factor])
                        new_coefficient * term_to_diff.coeff
                        
                        new_term = Term(new_dict, term_to_diff.q_deg, term_to_diff.N_deg, new_coefficient)
                        new_h_list = copy.deepcopy(term_to_diff.h_list)
                        new_term.h_list = new_h_list
                        new_term_collection.add_term(new_term)

                        # second contribution
                        new_dict = copy.deepcopy(term_to_diff.prod_dict)
                        add_to_dict(new_dict, ordind(factor[0], h[1]), 1)
                        add_to_dict(new_dict, ordind(factor[1], h[0]), 1)
                        add_to_dict(new_dict, factor, -1)
                

                        new_coefficient = CumulantPoly(((0, 0),), -term_to_diff.prod_dict[factor])
                        new_coefficient * term_to_diff.coeff
                        
                        new_term = Term(new_dict, term_to_diff.q_deg, term_to_diff.N_deg, new_coefficient)
                        new_h_list = copy.deepcopy(term_to_diff.h_list)
                        new_term.h_list = new_h_list
                        new_term_collection.add_term(new_term)

                        # third contribution, the contribution through the z
                        v = term_to_diff.get_fresh_index(0)
                        new_dict = copy.deepcopy(term_to_diff.prod_dict)
                        add_to_dict(new_dict, ordind(factor[0], v), 1)
                        add_to_dict(new_dict, ordind(factor[1],v), 1)
                        add_to_dict(new_dict, factor, -1)
                

                        new_coefficient = CumulantPoly(((0, 0),(random_edge_index, 1)), 4 * (term_to_diff.prod_dict[factor]))
                        new_coefficient * term_to_diff.coeff
                        
                        new_term = Term(new_dict, term_to_diff.q_deg, term_to_diff.N_deg, new_coefficient)
                        new_h_list = copy.deepcopy(term_to_diff.h_list)
                        new_term.h_list = new_h_list
                        new_term.add_h(h)
                        new_term_collection.add_term(new_term)

                # update the term_collection
                curr_term_collection = new_term_collection

                # store the latest computed results

                computed_curr_term_collections[diff_ind] = copy.deepcopy(curr_term_collection)

                # print(f"\n\nThe {diff_ind}th derivative:\n\n")
                # curr_term_collection.print_collection()
            
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

        
        # erase terms with left over h:s, since they will be negligible
        if throw_away_terms_with_remaining_hs:
            result_term_keys = copy.deepcopy(list(result_term_collection.terms.keys()))
            for result_term_key in result_term_keys:
                if len(result_term_collection.terms[result_term_key].h_list) > 0:
                    del result_term_collection.terms[result_term_key]

        return result_term_collection
 

def generate_term_obj_from_key(term_key):
    q_deg = term_key[0]
    N_deg = term_key[1]
    prod_dict = dict(term_key[2])
    coeff = CumulantPoly(((0, 0),), 1.0)
    term_obj = Term(prod_dict, q_deg, N_deg, coeff)
    term_obj.rearrange_vars()
    return term_obj

def load_term_collection(save_name) -> TermCollection:
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if f"{save_name}.pickle" in os.listdir(dir_path):
        print("Loading", os.path.join(dir_path, f"{save_name}.pickle"))
        with open(os.path.join(dir_path, f"{save_name}.pickle"), "rb") as f:
            term_collection = pickle.load(f)
    return term_collection



def type_ab_reduction():
    """
    Computes d/dt E[Im m(t, z(t))] and checks for cancellation
    """
    total_degree_limit = 100
    q_limit = 3
    degree_limit = 100
    a = 0
    b = 1
    reduce_fully = True

    
    m_term = Term({(2,2): 1}, 0, 0, CumulantPoly(((0, 0),), 1.0))
    # obtain the terms from partial/partial h_{ab} G_{vv}
    main_terms = m_term.expand_main_time_derivative_terms()
    
    main_terms.filter_and_group_equivalent(total_degree_limit, q_limit, degree_limit, reduce_fully)
    # main_terms.print_collection()
    
    # obtain the terms from d/dt z(t) * partial/partial z_(t) G_{vv}.
    edge_terms = m_term.get_edge_term()
    edge_terms.filter_and_group_equivalent(total_degree_limit, q_limit, degree_limit, reduce_fully)
    # edge_terms.print_collection()
    
    main_terms + edge_terms
    
    main_terms.filter_and_group_equivalent(total_degree_limit, q_limit, degree_limit, reduce_fully)

    # sanity check that we get reasonable terms 
    check_term_collection_for_strange_terms(main_terms, a, b)

    print(f"\n\nTerms to check for cancellation: ({len(main_terms.terms)} unique terms)\n\n")
    main_terms.print_collection(latex_format=True)
    
    # most computations are done in the following function
    reduce_to_true_type_0_term(main_terms, total_degree_limit, q_limit, degree_limit)

    
def list_contains_eq_term(a: list, term: Term):
    for other_term in a:
        if two_terms_equivalent(other_term, term):
            return True
    return False

def generate_identity_using_off_diag_expansion(start_term, factor_to_expand, total_degree_limit, q_degree_filter, degree_filter, switch_indices):

    term_collection = TermCollection()
    term_collection.add_term(start_term)

    term_collection.expand_off_diagonal_with_z_rule(start_term.get_key(), factor_to_expand, switch_indices)

    start_term.coeff * CumulantPoly(((0, 0),), -1.0)
    term_collection.add_term(start_term)

    check_term_collection_for_strange_terms(term_collection, 0, 1)

    term_collection.filter_and_group_equivalent(total_degree_limit, q_degree_filter, degree_filter)
    term_collection.mult_with_constant(CumulantPoly(((0,0),), 2.0))
    # print("\n\nIdentity generated:", "\n0 =\n\n")
    # term_collection.print_collection()
    return term_collection

def generate_identity_using_diag_expansion(start_term, total_degree_limit, q_degree_filter, degree_filter, reuse_index = None):

    term_collection = TermCollection()
    term_collection.add_term(start_term)
    if reuse_index is None:
        term_collection.add_m_with_z_rule(start_term.get_key())
    else:
        term_collection.add_m_with_z_rule(start_term.get_key(), reuse_index=reuse_index)

    start_term.coeff * CumulantPoly(((0, 0),), -1.0)
    term_collection.add_term(start_term)

    term_collection.filter_and_group_equivalent(total_degree_limit, q_degree_filter, degree_filter)

    term_collection.mult_with_constant(CumulantPoly(((0,0),), 2.0))
    # print("\n\nIdentity generated:", "\n0 =\n\n")
    # term_collection.print_collection()
    return term_collection


def generate_type_ab_terms(number_fresh_indices, print_out = False):
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
        print("\n\ntype_0_terms:", unique_type_0_terms)

    
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
        new_term = Term(new_prod_dict, 2, 1, CumulantPoly(((0,0),), 1.0))
        if not list_contains_eq_term(type_ab_terms, new_term):
            type_ab_terms.append(new_term)

        # ab ab aa bb * Type_0
        new_prod_dict = {}
        next_free_ind = 2
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[0] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[1] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, a, next_free_ind, ab_partition[2] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, b, b, next_free_ind, ab_partition[3] + 1)
        new_term = Term(new_prod_dict, 2, 1, CumulantPoly(((0,0),), 1.0))
        if not list_contains_eq_term(type_ab_terms, new_term):
            type_ab_terms.append(new_term)
        
        # ab ab ab ab * Type_0
        new_prod_dict = {}
        next_free_ind = 2
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[0] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[1] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[2] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, b, next_free_ind, ab_partition[3] + 1)
        new_term = Term(new_prod_dict, 2, 1, CumulantPoly(((0,0),), 1.0))
        if not list_contains_eq_term(type_ab_terms, new_term):
            type_ab_terms.append(new_term)

    type_a_terms = []
    for a_partition in a_partitions:
        # aa aa bb bb * Type_0
        new_prod_dict = {}
        next_free_ind = 1
        next_free_ind = add_loop_to_dict(new_prod_dict, a, a, next_free_ind, a_partition[0] + 1)
        next_free_ind = add_loop_to_dict(new_prod_dict, a, a, next_free_ind, a_partition[1] + 1)
        new_term = Term(new_prod_dict, 2, 1, CumulantPoly(((0,0),), 1.0))
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
    
    type_0_terms = [Term({}, 2, 1, CumulantPoly(((0,0),), 1.0))]
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
                    next_free_ind = add_loop_to_dict(new_term.prod_dict, next_free_ind, next_free_ind, next_free_ind+1, loop_size)
                assert not list_contains_eq_term(completed_terms, new_term)
                completed_terms.append(new_term)

    for term_ind in range(len(type_a_terms)):
        nbr_used_indices = type_a_terms[term_ind].get_nbr_vars()
        for unique_type_0_term in unique_type_0_terms:
            nbr_type_0_indices = sum(unique_type_0_term)
            if nbr_type_0_indices <= max_nbr_indices - nbr_used_indices:
                new_term = copy.deepcopy(type_a_terms[term_ind])
                for loop_size in unique_type_0_term:
                    next_free_ind = new_term.get_fresh_index(0)
                    next_free_ind = add_loop_to_dict(new_term.prod_dict, next_free_ind, next_free_ind, next_free_ind+1, loop_size)
                assert not list_contains_eq_term(completed_terms, new_term)
                completed_terms.append(new_term)

  
    nbr_used_indices = 0
    for unique_type_0_term in unique_type_0_terms:
        nbr_type_0_indices = sum(unique_type_0_term)
        if nbr_type_0_indices <= max_nbr_indices - nbr_used_indices:
            new_term = copy.deepcopy(type_0_terms[0])
            for loop_size in unique_type_0_term:
                next_free_ind = new_term.get_fresh_index(0)
                next_free_ind = add_loop_to_dict(new_term.prod_dict, next_free_ind, next_free_ind, next_free_ind+1, loop_size)
            assert not list_contains_eq_term(completed_terms, new_term)
            completed_terms.append(new_term)
    
    if print_out:
        print(f"\n\Completed terms: ({len(completed_terms)} terms unique terms)\n\n")
        for term in completed_terms:
            print(term)
            print(term.get_key())


    return completed_terms

def find_coeff_of_eq_term(a: TermCollection, term: Term, cumulant_term) -> float:
    for other_term_key in a.terms:
        other_term = a.terms[other_term_key]
        if two_terms_equivalent(other_term, term):
            if cumulant_term in other_term.coeff.poly:
                return other_term.coeff.poly[cumulant_term]
            else: 
                return 0.0
    return 0.0

def check_term_collection_for_strange_terms(term_collection: TermCollection, a: int, b: int):
    for term_key in term_collection.terms:
        term = term_collection.terms[term_key]
        var_occ, _ = term.get_var_occurances_and_indices()
        for var in var_occ:
            if var_occ[var] > 2 and (var != a and var != b):
                assert False, "Another index appears too many times."

def reduce_to_true_type_0_term(term_collection: TermCollection, total_degree_limit, q_degree_filter, degree_filter, non_trivial_identities = None):
    """
    Generates the identities as described in the article and builds the linear equation system and tries to solve it.
    """
    max_nbr_indices = 5
    a = 0
    b = 1

    identities_la = [] 
    # generate \Tau_4^{start} from the article
    type_ab_terms = generate_type_ab_terms(max_nbr_indices-3, print_out=True)
   
    for term in type_ab_terms:
        var_occ, _ = term.get_var_occurances_and_indices()
        # skip the terms of degree 0, since we only want terms of degree \geq 2 in our identities. 
        if term.get_degree() == 0:
            continue
        # print("Using start term ", term)
        for var in var_occ:
            if var_occ[var] > 2 and (var != a and var != b):
                assert False, "Another index appears too many times."

        # expansion rule 1:
        if len(term.get_var_indices()) <= max_nbr_indices-1:
            factors_to_expand = []
            for factor in term.prod_dict:
                if factor[0] != factor[1]:
                    factors_to_expand.append(factor)  
            for factor_to_expand in factors_to_expand:
                modified_term1 = copy.deepcopy(term) 
                modified_term2 = copy.deepcopy(term)                
                
                identities_la.append(generate_identity_using_off_diag_expansion(modified_term1, factor_to_expand, total_degree_limit, q_degree_filter, degree_filter, True))
                check_term_collection_for_strange_terms(identities_la[-1], a, b)
                identities_la.append(generate_identity_using_off_diag_expansion(modified_term2, factor_to_expand, total_degree_limit, q_degree_filter, degree_filter, False))
                check_term_collection_for_strange_terms(identities_la[-1], a, b)

        # expansion rule 2:
        if len(term.get_var_indices()) <= max_nbr_indices-2:
            modified_term1 = copy.deepcopy(term) 
            identities_la.append(generate_identity_using_diag_expansion(modified_term1, total_degree_limit, q_degree_filter, degree_filter))
            check_term_collection_for_strange_terms(identities_la[-1], a, b)
        
        #expansion rule 3:
        if a not in var_occ or var_occ[a] < 4:
            modified_term1 = copy.deepcopy(term) 
            identities_la.append(generate_identity_using_diag_expansion(modified_term1, total_degree_limit, q_degree_filter, degree_filter, reuse_index = a))
            check_term_collection_for_strange_terms(identities_la[-1], a, b)
        if b not in var_occ or var_occ[b] < 4:
            modified_term1 = copy.deepcopy(term) 
            identities_la.append(generate_identity_using_diag_expansion(modified_term1, total_degree_limit, q_degree_filter, degree_filter, reuse_index = b))
            check_term_collection_for_strange_terms(identities_la[-1], a, b)

    if non_trivial_identities is not None:
        for non_trivial_identity in non_trivial_identities:
            identities_la.append(non_trivial_identity)

    # the basis terms should be different if they have a different coefficient. And the identities have to be added once for each of the coefficients.
    occuring_cumulants = set()
            
    for term_key in term_collection.terms:
        term = term_collection.terms[term_key]
        for cumulant_term in term.coeff.poly:
            if term.coeff.poly[cumulant_term] != 0.0:
                occuring_cumulants.add(tuple(sorted(cumulant_term)))


    # loop through identities and term_collection to pick out the basis terms, this is regardless of cumulant coefficient
    basis_terms = []
    for identity in identities_la:
        for term_key in identity.terms:
            term = identity.terms[term_key]
            if not list_contains_eq_term(basis_terms, term):
                term_copy = copy.deepcopy(term)
                term_copy.coeff = CumulantPoly(((0,0),), 1.0)
                basis_terms.append(term_copy)

    for term_key in term_collection.terms:
        term = term_collection.terms[term_key]
        if not list_contains_eq_term(basis_terms, term):
            term_copy = copy.deepcopy(term)
            term_copy.coeff = CumulantPoly(((0,0),), 1.0)
            basis_terms.append(term_copy)


    print("\n\nBasis terms:\n\n")
    new_basis_terms = []
    basis_terms_cumulant_factors = []
    for i, term in enumerate(basis_terms):
        for cumulant_term in occuring_cumulants:
            term_copy = copy.deepcopy(term)
            term_copy.coeff = CumulantPoly(cumulant_term, 1.0)
            new_basis_terms.append(term_copy)
            basis_terms_cumulant_factors.append(cumulant_term)
            # print(new_basis_terms[-1], ",\\\\")
            print(new_basis_terms[-1], "basis term index", len(new_basis_terms)-1)

    basis_terms = new_basis_terms
    nbr_basis_terms = len(basis_terms)

    identity_vectors = set()
    for j in range(len(identities_la)):
        for cumulant_term in occuring_cumulants:
            term_collection_copy = copy.deepcopy(identities_la[j])
            term_collection_copy.mult_with_constant(CumulantPoly(cumulant_term, 1.0))
            identity_vector = []
            for i in range(nbr_basis_terms):
                identity_vector.append(find_coeff_of_eq_term(term_collection_copy, basis_terms[i], basis_terms_cumulant_factors[i]))
            identity_vector = tuple(identity_vector)
            identity_vectors.add(identity_vector)
    
    # if not check_zero:
    #     # we add the "basis vectors", representing the "true type-0 terms"
    #     for i in range(nbr_basis_terms):
    #         add_term = False
    #         if add_term:    
    #             new_vec = [0] * nbr_basis_terms
    #             new_vec[i] = 1
    #             identity_vectors.add(tuple(new_vec))
        
    # span_matrix is the coefficient matrix of the linear equation system
    span_matrix = np.zeros((nbr_basis_terms, len(identity_vectors)), dtype=object)
    for i in range(nbr_basis_terms):
        for j, identity_vector in enumerate(identity_vectors):
            assert abs(int(identity_vector[i]) - identity_vector[i]) < 1e-8, "Should be integers"
            span_matrix[i][j] = Fraction(int(identity_vector[i]))
    
    # multiply by 1/2 since the identities are multiplied by 2 when they are generated
    span_matrix *= Fraction(1, 2)
    
    print("Coefficient matrix is of dimension", nbr_basis_terms, "x", len(identity_vectors))
    print("Coefficient matrix:")

    for j in range(span_matrix.shape[1]):
        non_zero_elements = []
        for i in range(span_matrix.shape[0]):
            if span_matrix[i][j] != 0:
                non_zero_elements.append((i+1, span_matrix[i][j]))
        # sort the non-zero elements by their row index
        non_zero_elements = sorted(non_zero_elements, key=lambda x: x[0])
        non_zero_elements = [f"{x[0]}th entry: {str(x[1])}" for x in non_zero_elements]
        print(f"{j+1}th column of coefficient matrix: {', '.join(non_zero_elements)}")

    # print("Span matrix:\n", span_matrix)
    term_collection_vector = np.zeros(nbr_basis_terms, dtype=object)
    for i in range(nbr_basis_terms):
        term_collection_vector[i] = Fraction(int(find_coeff_of_eq_term(term_collection, basis_terms[i], basis_terms_cumulant_factors[i])))
    
    
    print("Rhs of linear equation system:")

    non_zero_elements = []
    for i in range(term_collection_vector.shape[0]):
        if term_collection_vector[i] != 0:
            non_zero_elements.append((i+1, term_collection_vector[i]))
    
    non_zero_elements = sorted(non_zero_elements, key=lambda x: x[0])
    non_zero_elements = [f"{x[0]}th entry: {str(x[1])}" for x in non_zero_elements]
    print(f"d/dt E[m(t, z(t))] on vector form: {', '.join(non_zero_elements)}")

    # x_lst_sqr, _, _, _ = np.linalg.lstsq(span_matrix, term_collection_vector, rcond=None)
    x_frac, rank = gaussian_elim.solveLinear(span_matrix, term_collection_vector)

    if rank == -1:
        print("Term_collection NOT 0! Cancellation not obtained!")
        return
    print("Exact solution in Fractions:", x_frac, "Rank:", rank)
    # x_rounded = x_lst_sqr.round()
    # print("Rounded:", x_rounded)
    print("Largest absolute value in solution:", max(abs(x_frac)))
    remainder = term_collection_vector - span_matrix @ x_frac
    print("Number of non-zero terms:", sum([0 if x == Fraction(0, 1) else 1 for x in x_frac]))
    print("Remainder:", remainder)
    
    remainder_equal_zero = True
    for i in range(nbr_basis_terms):
        if remainder[i] != Fraction(0, 1):
            remainder_equal_zero = False

    if remainder_equal_zero:
        print("Cancellation obtained!")
    else:
        print("Term_collection NOT 0! Cancellation not obtained!")
        remainder = span_matrix @ x_frac - term_collection_vector
        for i in range(nbr_basis_terms):
            if np.abs(remainder[i]) > 1e-9:
                print("Basis term", basis_terms[i], basis_terms[i].get_key(), "\nnot reduced, remaining entry:", remainder[i])
        return
    
    non_zero_elements = []
    for i in range(x_frac.shape[0]):
        if x_frac[i] != 0:
            non_zero_elements.append((i+1, x_frac[i]))
    
    non_zero_elements = sorted(non_zero_elements, key=lambda x: x[0])
    non_zero_elements = [f"{x[0]}th entry: {str(x[1])}" for x in non_zero_elements]
    print(f"Solution to linear equation system: {', '.join(non_zero_elements)}")

    # prints the identities used to express d/dt E[Im m(t, z(t))] as a linear combination of the identities
    for c in range(span_matrix.shape[1]):
        if x_frac[c] != Fraction(0, 1):
            print("\n\nUsed identity:")
            print(f"\n\n{x_frac[c]} * (")
            for r in range(span_matrix.shape[0]):
                if span_matrix[r][c] == 0:
                    continue
                term_copy = copy.deepcopy(basis_terms[r])
                term_copy.coeff * CumulantPoly(((0,0),), span_matrix[r][c])
                print(term_copy, "basis term index", r)
            print(")")

if __name__ == "__main__":
    type_ab_reduction()
