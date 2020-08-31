from TREE.tree import Tree
import numpy as np
import unittest
from scipy.stats import pearsonr

class TestTree(unittest.TestCase):
    def test_numbering(self):
        tree = Tree()
        kappa = np.arange(8)+0.1
        tree.append(kappa.reshape(-1, 4), [0, 1])
        xi = 0.5
        tree.choose_event(xi)
        cumsum = np.cumsum(kappa)/np.sum(kappa)
        jump_id = np.sum(cumsum<xi)%4
        index = int((np.sum(cumsum<xi)+1)/4)
        self.assertEqual(tree.get_jump_id(), jump_id)
        self.assertEqual(tree.get_index(), index)

    def test_large(self):
        tree = Tree()
        n_atoms = int(1e4)
        kappa = np.zeros(n_atoms*4)+1.0e-4
        index = 12
        jump_id = 3
        kappa[index*4+jump_id] = 1e4
        tree.append(kappa.reshape(-1, 4), np.arange(n_atoms))
        tree.choose_event(0.5)
        self.assertEqual(tree.get_jump_id(), jump_id)
        self.assertEqual(tree.get_index(), index)

    def test_remove(self):
        tree = Tree()
        n_atoms = 4
        kappa = np.zeros(n_atoms*4)+1.0e-4
        index = 1
        jump_id = 3
        kappa[index*4+jump_id] = 1e4
        tree.append(kappa.reshape(-1, 4), np.arange(n_atoms))
        tree.choose_event(0.5)
        tree.remove()
        tree.append([np.ones(4)], [1])
        tree.choose_event(0.4)
        self.assertEqual(tree.get_index(), 1)
        self.assertEqual(tree.get_jump_id(), 1)

    def test_recursive(self):
        tree = Tree()
        n_atoms = int(1e4)
        kappa = np.random.rand(n_atoms*4).reshape(-1, 4)
        tree.append(kappa, np.arange(n_atoms))
        for _ in range(100):
            tree.choose_event(np.random.rand())
            index = tree.get_index()
            kappa[index] = np.random.rand()
            tree.update_kappa(kappa[index])
        self.assertAlmostEqual(tree.get_kappa(), np.sum(kappa))

    def test_recursive_remove(self):
        tree = Tree()
        n_atoms = int(1e4)
        kappa = np.random.rand(n_atoms*4).reshape(-1, 4)
        tree.append(kappa, np.arange(n_atoms))
        for _ in range(100):
            tree.choose_event(np.random.rand())
            index = tree.get_index()
            tree.remove()
            kappa[index] = np.random.rand()
            tree.append([kappa[index]], [index])
        self.assertAlmostEqual(tree.get_kappa(), np.sum(kappa))

    def test_pearson(self):
        n_atoms = 100
        tree = Tree()
        kappa = np.random.rand(n_atoms*4).reshape(-1, 4)
        index_lst = []
        tree.append(kappa, np.arange(n_atoms))
        for _ in range(1_000_000):
            tree.choose_event(np.random.rand())
            index = tree.get_index()
            index_lst.append([tree.get_index(), tree.get_jump_id()])
            tree.update_kappa(kappa[index])
        index_lst = np.array(index_lst)
        ind, count = np.unique(index_lst[:,0]*4+index_lst[:,1], return_counts=True)
        count = count/np.sum(count)
        p = kappa.flatten()[ind]/np.sum(kappa.flatten()[ind])
        self.assertGreater(pearsonr(count, p)[0], 0.99)

if __name__ == "__main__":
    unittest.main()
