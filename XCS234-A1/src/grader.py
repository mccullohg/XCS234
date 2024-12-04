#!/usr/bin/env python3
import unittest
import random
import argparse
import inspect
from graderUtil import graded, CourseTestRunner, GradedTestCase
import numpy as np

# Import student submission
from randommdp import RandomMDP
from riverswim import RiverSwim

import submission


#########
# TESTS #
#########

class BaseTest(GradedTestCase):
    def setUp(self):
        self.seed = 5678
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.num_test_mdps = 100
        self.rswim_weak_test = RiverSwim('WEAK')
        self.rswim_weak = RiverSwim('WEAK', self.seed)
        self.rswim_medium = RiverSwim('MEDIUM', self.seed)
        self.rswim_strong = RiverSwim('STRONG', self.seed)
        self.rswim_envs = [self.rswim_weak, self.rswim_medium, self.rswim_strong]
        self.rswim_env_names = ['WEAK', 'MEDIUM', 'STRONG']
        self.gammas = [[0.67, 0.68, 0.99], [0.77, 0.78], [0.93, 0.94]]
        self.tol = 0.001

class Test_1a(BaseTest):
    def setUp(self):
        super().setUp()
        self.state = 0
        self.action = 0
        self.R = np.array([[1, 2], [3, 4]])
        self.T = np.array([
            [[0.5, 0.5], [0.8, 0.2]],
            [[0.2, 0.8], [0.3, 0.7]]
        ])
        self.gamma = 0.9
        self.V = np.array([10, 20])

    @graded()
    def test_0(self):
        """1a-0-basic: Bellman backup expected type"""
        backup_val = submission.bellman_backup(self.state, self.action, self.R, self.T, self.gamma, self.V)
        self.assertIsInstance(backup_val, float, msg=f"Expected type float but got {type(backup_val).__name__}")

    @graded()
    def test_1(self):
        """1a-1-basic: Bellman backup expected value"""
        expected_backup_val = 14.5
        backup_val = submission.bellman_backup(self.state, self.action, self.R, self.T, self.gamma, self.V)
        self.assertAlmostEqual(backup_val, expected_backup_val, delta=self.tol, msg=f"Expected {expected_backup_val} but got {backup_val}")

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_1b(BaseTest):
    def setUp(self):
        super().setUp()
        self.R = np.array([[1, 2], [3, 4]])
        self.T = np.array([
            [[0.5, 0.5], [0.8, 0.2]],
            [[0.2, 0.8], [0.3, 0.7]]
        ])
    
    @graded()
    def test_0(self):
        """1b-0-basic: Policy evaluation expected type and shape"""
        policy = np.array([0, 1])
        gamma = 0.9
        value_function = submission.policy_evaluation(policy, self.R, self.T, gamma)
        expected_shape = (2,)
        self.assertIsInstance(value_function, np.ndarray, msg=f"Expected type np.ndarray but got {type(value_function).__name__}")
        self.assertEqual(
            value_function.shape, expected_shape,
            msg=f"Expected shape {expected_shape} but got {value_function.shape}"
        )
    
    @graded()
    def test_1(self):
        """1b-1-basic: Policy evaluation expected value"""
        policy = np.array([0, 1])
        gamma = 0.9
        value_function = submission.policy_evaluation(policy, self.R, self.T, gamma)
        expected_value_function = np.array([26.455, 30.113])
        self.assertTrue(
            np.allclose(value_function, expected_value_function, atol=self.tol),
            msg=f"Expected {expected_value_function} but got {value_function}"
        )    

    @graded()
    def test_2(self):
        """1b-2-basic: Policy evaluation when gamma is zero"""
        policy = np.array([0, 1])
        gamma = 0.0
        value_function = submission.policy_evaluation(policy, self.R, self.T, gamma)
        expected_value_function = np.array([self.R[0, 0], self.R[1, 1]])
        self.assertTrue(
            np.allclose(value_function, expected_value_function, atol=self.tol),
            msg=f"Policy evaluation with gamma=0 did not return immediate rewards. Expected {expected_value_function} but got {value_function}."
        )
    
    @graded()
    def test_3(self):
        """1b-3-basic: Policy evaluation when all rewards are zero"""
        R = np.zeros((2, 2))
        policy = np.array([0, 1])
        gamma = 0.9
        value_function = submission.policy_evaluation(policy, R, self.T, gamma)
        expected_value_function = np.zeros(2)
        self.assertTrue(
            np.allclose(value_function, expected_value_function, atol=self.tol), 
            msg=f"Policy evaluation with zero rewards did not return a zero value function. Expected {expected_value_function} but got {value_function}."
        )
        
    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_1c(BaseTest):
    def setUp(self):
        super().setUp()
    
    @graded()
    def test_0(self):
        """1c-0-basic: Policy improvement expected type and shape"""
        R = np.array([[10, 0], [0, 5]])
        T = np.array([[[1, 0], [0, 1]],
                   [[1, 0], [0, 1]]])
        V_policy = np.array([10, 20])
        gamma = 0.9
        expected_shape = (2,)
        improved_policy = submission.policy_improvement(R, T, V_policy, gamma)
        self.assertIsInstance(improved_policy, np.ndarray, msg=f"Expected type np.ndarray but got {type(improved_policy).__name__}")
        self.assertEqual(
            improved_policy.shape, expected_shape,
            msg=f"Expected shape {expected_shape} but got {improved_policy.shape}"
        )
    
    @graded()
    def test_1(self):
        """1c-1-basic: Policy improvement expected policy does not change"""
        R = np.array([[1, 2], [2, 1]])
        T = np.array([[[0.5, 0.5], [0.5, 0.5]], 
                   [[0.5, 0.5], [0.5, 0.5]]])
        V_policy = np.array([10, 20])
        gamma = 0.5
        new_policy = submission.policy_improvement(R, T, V_policy, gamma)
        expected_policy = np.array([1, 0])
        assert np.array_equal(new_policy, expected_policy), f"Expected {expected_policy} but got {new_policy}"
    
    @graded()
    def test_2(self):
        """1c-2-basic: Policy improvement selects optimal policy"""
        R = np.array([[0, 10], [10, 0]])
        T = np.array([[[1, 0], [0, 1]], 
                   [[1, 0], [0, 1]]])
        V_policy = np.array([5, 5])
        gamma = 0.9
        new_policy = submission.policy_improvement(R, T, V_policy, gamma)
        expected_policy = np.array([1, 0])
        assert np.array_equal(new_policy, expected_policy), f"Expected {expected_policy} but got {new_policy}"
    
    ### BEGIN_HIDE ###
    ### END_HIDE ###

class Test_1d(BaseTest):
    def setUp(self):
        super().setUp()

    @graded()
    def test_0(self):
        """1d-0-basic: Policy iteration expected types and shapes"""
        R = np.array([[10, 0], [0, 5]])
        T = np.array([[[1, 0], [0, 1]],
                   [[1, 0], [0, 1]]])
        policy = np.array([0, 1])
        gamma = 0.9
        expected_shape = (2,)
        V_policy, policy = submission.policy_iteration(R, T, gamma)
        self.assertIsInstance(V_policy, np.ndarray, msg=f"Expected type np.ndarray but got {type(V_policy).__name__}")
        self.assertIsInstance(policy, np.ndarray, msg=f"Expected type np.ndarray but got {type(policy).__name__}")
        self.assertEqual(
            V_policy.shape, expected_shape,
            msg=f"Expected shape {expected_shape} but got {V_policy.shape} for value function"
        )
        self.assertEqual(
            policy.shape, expected_shape,
            msg=f"Expected shape {expected_shape} but got {policy.shape} for policy"
        )

    @graded()
    def test_1(self):
        """1d-1-basic: Policy iteration expected value function and policy"""
        R = np.array([[1, 0], [0, 2]])
        T = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]])
        gamma = 0.9
        V_policy, policy = submission.policy_iteration(R, T, gamma)
        expected_V_policy = np.array([9.991, 2.0])
        expected_policy = np.array([0, 1])

        assert np.array_equal(policy, expected_policy), f"Expected {expected_policy} but got {policy}"
        assert np.allclose(V_policy, expected_V_policy, atol=self.tol), f"Expected {expected_V_policy} but got {V_policy}"
    
    @graded()
    def test_2(self):
        """1d-2-basic: Policy iteration with zero rewards"""
        R = np.zeros((3, 2))
        T = np.array([[[0.5, 0.5, 0], [0, 0, 1]],
                      [[0.3, 0.7, 0], [0, 0, 1]],
                      [[0.1, 0.9, 0], [0, 0, 1]]])
        gamma = 0.9
        V_policy, policy = submission.policy_iteration(R, T, gamma)
        expected_V_policy = np.zeros(3)
        expected_policy = np.zeros(3, dtype=int)

        assert np.array_equal(policy, expected_policy), f"Expected {expected_policy} but got {policy}"
        assert np.allclose(V_policy, expected_V_policy, atol=self.tol), f"Expected {expected_V_policy} but got {V_policy}"
    
    @graded()
    def test_3(self):
        """1d-3-basic: Policy iteration with gamma=0.99 and WEAK current"""
        R, T = self.rswim_weak_test.get_model()
        gamma = 0.99
        tol = 0.001
        V_policy, policy = submission.policy_iteration(R, T, gamma, tol)
        expected_V_policy = np.array([30.693, 31.211, 32.382, 33.78, 35.29,  36.881])
        expected_policy = np.array([1, 1, 1, 1, 1, 1])
        assert np.array_equal(policy, expected_policy), f"Expected {expected_policy} but got {policy}"
        assert np.allclose(V_policy, expected_V_policy, atol=self.tol), f"Expected {expected_V_policy} but got {V_policy}"

    ### BEGIN_HIDE ###
    ### END_HIDE ###


class Test_1e(BaseTest):
    def setUp(self):
        super().setUp()
    
    @graded()
    def test_0(self):
        """1e-0-basic: Value iteration expected types and shapes"""
        R = np.array([[10, 0], [0, 5]])
        T = np.array([[[1, 0], [0, 1]],
                   [[1, 0], [0, 1]]])
        gamma = 0.9
        expected_shape = (2,)
        V, policy = submission.value_iteration(R, T, gamma)
        self.assertIsInstance(V, np.ndarray, msg=f"Expected type np.ndarray but got {type(V).__name__}")
        self.assertIsInstance(policy, np.ndarray, msg=f"Expected type np.ndarray but got {type(policy).__name__}")
        self.assertEqual(
            V.shape, expected_shape,
            msg=f"Expected shape {expected_shape} but got {V.shape} for value function"
        )
        self.assertEqual(
            policy.shape, expected_shape,
            msg=f"Expected shape {expected_shape} but got {policy.shape} for policy"
        )


    
    @graded()
    def test_1(self):
        """1e-1-basic: Value iteration expected value function and policy"""
        R = np.array([[1, 0], [0, 2]])
        T = np.array([[[1, 0], [0, 0]], [[0, 1], [0, 0]]])
        gamma = 0.9
        V, policy = submission.value_iteration(R, T, gamma)
        expected_V = np.array([9.991, 2.0])
        expected_policy = np.array([0, 1])

        assert np.array_equal(policy, expected_policy), f"Expected {expected_policy} but got {policy}"
        assert np.allclose(V, expected_V, atol=self.tol), f"Expected {expected_V} but got {V}"
    
    @graded()
    def test_2(self):
        """1e-2-basic: Value iteration with zero rewards"""
        R = np.zeros((3, 2))
        T = np.array([[[0.5, 0.5, 0], [0, 0, 1]],
                      [[0.3, 0.7, 0], [0, 0, 1]],
                      [[0.1, 0.9, 0], [0, 0, 1]]])
        gamma = 0.9
        V, policy = submission.value_iteration(R, T, gamma)
        expected_V = np.zeros(3)
        expected_policy = np.zeros(3, dtype=int)

        assert np.array_equal(policy, expected_policy), f"Expected {expected_policy} but got {policy}"
        assert np.allclose(V, expected_V, atol=self.tol), f"Expected {expected_V} but got {V}"
    
    @graded()
    def test_3(self):
        """1e-3-basic: Value iteration with gamma=0.99 and WEAK current"""
        R, T = self.rswim_weak_test.get_model()
        gamma = 0.99
        tol = 0.001
        V, policy = submission.value_iteration(R, T, gamma, tol)
        expected_V = np.array([30.693, 31.211, 32.382, 33.78, 35.29,  36.881])
        expected_policy = np.array([1, 1, 1, 1, 1, 1])
        assert np.array_equal(policy, expected_policy), f"Expected {expected_policy} but got {policy}"
        assert np.allclose(V, expected_V, atol=self.tol), f"Expected {expected_V} but got {V}"
    
    ### BEGIN_HIDE ###
    ### END_HIDE ###


def getTestCaseForTestID(test_id):
    question, part, _ = test_id.split("-")
    g = globals().copy()
    for name, obj in g.items():
        if inspect.isclass(obj) and name == ("Test_" + question):
            return obj("test_" + part)


if __name__ == "__main__":
    # Parse for a specific test
    parser = argparse.ArgumentParser()
    parser.add_argument("test_case", nargs="?", default="all")
    test_id = parser.parse_args().test_case

    assignment = unittest.TestSuite()
    if test_id != "all":
        assignment.addTest(getTestCaseForTestID(test_id))
    else:
        assignment.addTests(
            unittest.defaultTestLoader.discover(".", pattern="grader.py")
        )
    CourseTestRunner().run(assignment)
