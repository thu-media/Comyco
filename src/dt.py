from sklearn import tree
import sklearn
import numpy as np
import pydotplus
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from sklearn.tree import _tree

class ActorNetwork(object):
    def __init__(self, state_dim, action_dim, max_depth):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.max_depth = max_depth
        self.clf = tree.DecisionTreeClassifier(max_depth=self.max_depth)

    def train(self, pred, target):
        pred = np.reshape(pred, [-1, self.s_dim[0]])
        target = np.argmax(target, axis=1)
        self.clf = self.clf.fit(pred, target)

    def predict(self, inputs):
        inputs = np.reshape(inputs, [-1, self.s_dim[0]])
        # sklearn.utils.validation.check_is_fitted(self.clf, 'tree_')
        try:
            return self.clf.predict_proba(inputs)[0]
        except:
            output = np.zeros(self.a_dim)
            output[0] = 1.
            return output

    def save(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.clf, f)
        f.close()

    def load(self, filename):
        f = open(filename, 'rb')
        self.clf = pickle.load(f)
        f.close()

    def export(self, filename):
        dot_data = StringIO()
        export_graphviz(self.clf, out_file=dot_data, filled=True)
        out_graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        out_graph.write_svg('tree/' + filename + '.svg')
        
    def compute_entropy(self, x):
        """
        Given vector x, computes the entropy
        H(x) = - sum( p * log(p))
        """
        H = 0.0
        x = np.clip(x, 1e-5, 1.)
        for i in range(len(x)):
            if 0 < x[i] < 1:
                H -= x[i] * np.log(x[i])
        return H

    def tree_to_code(self, feature_names, filename):
        tree_ = self.clf.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]
        f = open(filename, 'w')
        f.write("def predict({}):".format(", ".join(feature_names)))
        f.write('\n')
        def recurse(node, depth):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]
                f.write("{}if {} <= {}:".format(indent, name, threshold))
                f.write('\n')
                recurse(tree_.children_left[node], depth + 1)
                f.write("{}else:  # if {} > {}".format(indent, name, threshold))
                f.write('\n')
                recurse(tree_.children_right[node], depth + 1)
            else:
                tmpstr = '['
                for p in tree_.value[node][0]:
                    tmpstr += str(int(p))
                    tmpstr += ', '
                tmpstr += ']'
                tmpstr = tmpstr.replace(', ]', ']')
                f.write("{}return {}".format(indent, tmpstr))
                f.write('\n')

        recurse(0, 1)
        f.close()