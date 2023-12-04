"""
Author      : Jackey Weng
Student ID  : 40130001
Description : Assignment 2
"""


class RNA:
    def __init__(self, sequence, probability_list, activity_level):
        # Set each nucleotide as an attribute/feature of the RNA sequence
        for i, nucleotide in enumerate(sequence, start=1):
            setattr(self, f"nucleotide_{i}", nucleotide)

        # Set each probability of the upper triangle as an attribute/feature
        for i, prob in enumerate(probability_list, start=1):
            setattr(self, f"bppm_prob_{i}", prob)

        # self.probability_matrix = probability_matrix
        self.activity_level = activity_level

    # Return a dictionary of the rna attributes
    def to_dictionary(self):
        return self.__dict__
