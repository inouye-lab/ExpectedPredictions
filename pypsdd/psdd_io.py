from .sdd import SddNode
from .psdd import PSddNode
import math

import numpy as np
# AC: TODO: check vtree scope

########################################
# UTILITY FUNCTIONS
########################################

def pairs(lst):
    """A generator for (prime,sub) pairs"""
    if lst is None: return
    it = iter(lst)
    for x in it:
        y = it.next()
        yield (x,y)

########################################
# SDD I/O
########################################

_sdd_file_header = \
    ("c ids of sdd nodes start at 0\n"
     "c sdd nodes appear bottom-up, children before parents\n"
     "c\n"
     "c file syntax:\n"
     "c sdd count-of-sdd-nodes\n"
     "c F id-of-false-sdd-node\n"
     "c T id-of-true-sdd-node\n"
     "c L id-of-literal-sdd-node id-of-vtree literal\n"
     "c D id-of-decomposition-sdd-node id-of-vtree"
     " number-of-elements {id-of-prime id-of-sub}*\n"
     "c\n")

def sdd_read(filename,manager):
    """Read an SDD from file"""
    vtree_nodes = manager.vtree.to_list()
    f = open(filename,'r')
    for line in f:
        node = None
        if line.startswith('c'): continue
        elif line.startswith('sdd'):
            node_count = int(line[3:])
            nodes = [None]*node_count
        elif line.startswith('F'):
            node_id = int(line[2:])
            node = manager.false
        elif line.startswith('T'):
            node_id = int(line[2:])
            node = manager.true
        elif line.startswith('L'):
            node_id,vtree_id,lit = [ int(x) for x in line[2:].split() ]
            node = manager.literals[lit]
        elif line.startswith('D'):
            line = [ int(x) for x in line[2:].split() ]
            node_id,vtree_id,size = line[:3]
            elements = [ nodes[my_id] for my_id in line[3:] ]
            elements = [ element for element in pairs(elements) ]
            vtree_node = vtree_nodes[vtree_id]
            node = manager.lookup_node(elements,vtree_node)
        if node:
            nodes[node_id] = node
    f.close()
    return node

def _set_index(root):
    """set up index for saving SDD to file"""
    for index,node in enumerate(root.pre_order()):
        node.index = index

def sdd_save(root,filename):
    """Save an SDD to file"""
    _set_index(root)
    with open(filename,'w') as f:
        f.write(_sdd_file_header)
        f.write('sdd %d\n' % root._node_count())
        for node in root:
            f.write('%s\n' % node.__repr__(use_index=True))

_dot_node_fmt = \
    ('\nn%u [label="%u",style=filled,fillcolor=gray95,'
     'shape=circle,fixedsize=true,height=.5,width=.5];')
_dot_element_fmt = \
    ('\nn%ue%u\n'
     '[label=<\n'
     '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="0" PORT="T">\n'
     '<TR>\n<TD PORT="L">%s</TD>\n<TD PORT="R">%s</TD>\n</TR>\n'
     '</TABLE>\n>,\n'
     'shape=none,margin=0,fillcolor=white,style=filled,\n'
     'fontsize=20,fontname="Times",\n'
     '];\n')
_dot_or_fmt = '\nn%u->n%ue%u:T [arrowsize=.50];'
_dot_box_fmt = \
    ('\nn%ue%u:%s:c->n%u '
     '[arrowsize=.50,tailclip=false,arrowtail=dot,dir=both];')
_dot_terminal_fmt = '\nn%u [label="%s",shape=box];'
_dot_names = "0ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def sdd_save_as_dot(root,filename):
    _set_index(root)

    # open, write header
    f = open(filename,'w')
    f.write('digraph sdd {\n')
    f.write('overlap=false\n')
    if not root.is_decomposition():
        _print_terminal_sdd_as_dot(root,f)
    _print_node_ranks(root,f)

    # write nodes
    for n in root.pre_order():
        if not n.is_decomposition(): continue
        f.write(_dot_node_fmt % (n.index,n.vtree.id))

        for i,(p,s) in enumerate(n.elements):
            p_label,s_label = _node_label(p),_node_label(s)
            f.write(_dot_element_fmt % (n.index,i,p_label,s_label))
            f.write(_dot_or_fmt % (n.index,n.index,i))
            if p.is_decomposition(): 
                f.write(_dot_box_fmt % (n.index,i,'L',p.index))
            if s.is_decomposition():
                f.write(_dot_box_fmt % (n.index,i,'R',s.index))

    f.write('\n\n\n}\n')
    f.close()

def _print_node_ranks(root,f):
    pass

def _print_terminal_sdd_as_dot(root,f):
    label = _node_label(root)
    f.write(_dot_terminal_fmt % (root.index,label))

def _node_label(root,labels=_dot_names):
    """return/create symbol for terminal SDD"""
    if root.is_false():     return "&#8869;"
    elif root.is_true():    return "&#8868;"
    elif root.is_literal(): return _literal_label(root.literal,labels=labels)
    else: return "   "

def _literal_label(literal,labels=_dot_names):
    """return/create symbol for literal"""
    var = abs(literal)
    neg_label = "&not;" if literal < 0 else ""
    var_label = labels[var] if var <= 26 else str(var)
    return "%s<I>%s</I>" % (neg_label,var_label)

########################################
# PSDD I/O
########################################

_psdd_file_header = \
    ("c ids of psdd nodes start at 0\n"
     "c psdd nodes appear bottom-up, children before parents\n"
     "c\n"
     "c file syntax:\n"
     "c psdd count-of-sdd-nodes\n"
     "c L id-of-literal-sdd-node id-of-vtree literal\n"
     "c T id-of-trueNode-sdd-node id-of-vtree variable log(litProb)\n"
     "c D id-of-decomposition-sdd-node id-of-vtree"
     " number-of-elements {id-of-prime id-of-sub log(elementProb)}*\n"
     "c\n")

_dot_psdd_node_fmt = \
    ('\nn%u [label="%u",style=filled,fillcolor=gray95,'
     'shape=circle,fixedsize=true,height=.5,width=.5,color=%s,xlabel="%s"];')
_dot_psdd_element_fmt = \
    ('\nn%ue%u\n'
     '[label=<\n'
     '<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="0" PORT="T" COLOR="%s">\n'
     '<TR>\n<TD PORT="L">%s</TD>\n<TD PORT="R">%s</TD>\n</TR>\n'
     '</TABLE>\n>,\n'
     'shape=none,margin=0,fillcolor=white,style=filled,\n'
     'fontsize=20,fontname="Times",\n'
     '];\n')
_dot_psdd_or_fmt = '\nn%u->n%ue%u:T [arrowsize=.50,label=<%s>,color=%s];'
_dot_psdd_box_fmt = \
    ('\nn%ue%u:%s:c->n%u '
     '[arrowsize=.50,tailclip=false,arrowtail=dot,dir=both,color=%s];')


def psdd_save_as_dot(root,filename,subcircuit=None,labels=_dot_names):
    _set_index(root)

    # open, write header
    f = open(filename,'w')
    f.write('digraph sdd {\n')
    f.write('overlap=false\n')
    if not root.is_decomposition():
        _print_terminal_sdd_as_dot(root,f)
    _print_node_ranks(root,f)

    if subcircuit: subcircuit.probability()

    # write nodes
    for n in root.pre_order():
        if not n.is_decomposition(): continue
        n_on_sc = False
        if subcircuit:
            sc_node = subcircuit.node_of_vtree(n.vtree)
            n_on_sc = sc_node.node == n
        node_color = "red" if n_on_sc else "black" 
        pr_label = "" if n_on_sc else "" # AC

        f.write(_dot_psdd_node_fmt % (n.index,n.vtree.id,node_color,pr_label))
        for i,(p,s) in enumerate(n.elements):
            element = (p,s)
            el_on_sc = n_on_sc and sc_node.element == element
            el_color = "red" if el_on_sc else "black"
            p_label = _psdd_node_label(p,subcircuit=subcircuit,labels=labels)
            s_label = _psdd_node_label(s,subcircuit=subcircuit,labels=labels)

            if n.is_false_sdd or not hasattr(n,'theta') or element not in n.theta:
                edge_label = ''
            elif n.theta[(p,s)] == n.theta_sum:
                edge_label = '1'
            else:
                edge_label = "%.2f" % (n.theta[(p,s)]/n.theta_sum)

            f.write(_dot_psdd_element_fmt % (n.index,i,el_color,p_label,s_label))
            f.write(_dot_psdd_or_fmt % (n.index,n.index,i,edge_label,el_color))
            if p.is_decomposition(): 
                f.write(_dot_psdd_box_fmt % (n.index,i,'L',p.index,el_color))
            if s.is_decomposition():
                f.write(_dot_psdd_box_fmt % (n.index,i,'R',s.index,el_color))

    f.write('\n\n\n}\n')
    f.close()

def _psdd_node_label(root,subcircuit=None,labels=_dot_names):
    """return/create symbol for terminal SDD"""
    if root.is_true():
        var = root.vtree.var
        label = _literal_label(var,labels=labels)
        if subcircuit:
            sc_node = subcircuit.node_of_vtree(root.vtree)
        if subcircuit and root == sc_node.node:
            theta = "%.2f" % (root.theta[sc_node.element]/root.theta_sum)
            label = label if sc_node.element else "&not;" + label
        else:
            if hasattr(root,'theta'):
                theta = "%.2f" % (root.theta[1]/root.theta_sum)
            else:
                theta = "."
        return "%s:%s" % (label,theta)
    else: return _node_label(root,labels=labels)

########################################
# PSDD I/O (YITAO)
########################################

_psdd_yitao_file_header = \
    ("c ids of psdd nodes start at 0\n"
     "c psdd nodes appear bottom-up, children before parents\n"
     "c\n"
     "c file syntax:\n"
     "c psdd count-of-sdd-nodes\n"
     "c L id-of-literal-sdd-node id-of-vtree literal\n"
     "c T id-of-trueNode-sdd-node id-of-vtree variable log(litProb)\n"
     "c D id-of-decomposition-sdd-node id-of-vtree number-of-elements {id-of-prime id-of-sub log(elementProb)}*\n"
     "c\n")

def psdd_yitao_read(filename,pmanager):
    """Read a PSDD (Yitao) from file (not well tested!!!)"""
    vtree_nodes = pmanager.vtree.to_list()
    var_to_vtree = pmanager.vtree.var_to_vtree()
    f = open(filename,'r')
    for line in f:
        node = None
        if line.startswith('c'): continue
        if line.startswith('psdd'):
            node_count = int(line[4:]) # ignored
            nodes = {}
        elif line.startswith('F'): # no FALSE nodes
            pass
        elif line.startswith('T'):
            line = line[2:].split()
            node_id,vtree_id,var = [ int(x) for x in line[:-1] ]
            theta = np.float64(line[-1])
            vtree = var_to_vtree[var]
            node = PSddNode(SddNode.TRUE,None,vtree,pmanager)
            node.theta = [ np.float64(1.0)-np.exp(theta), np.exp(theta) ]
            node.theta_sum = sum(node.theta)
        elif line.startswith('L'):
            node_id,vtree_id,lit = [ int(x) for x in line[2:].split() ]
            node = pmanager.literals[lit]
            node.theta = [np.float64(0.0),np.float64(0.0)]
            node.theta[node.literal > 0] = np.float64(1.0)
            node.theta_sum = np.float64(1.0)
        elif line.startswith('D'):
            line = line[2:].split()
            node_id,vtree_id,size = [ int(x) for x in line[:3] ]
            line_iter = iter(line[3:])
            elements,theta = list(),dict()
            for i in range(size):
                p = nodes[int(line_iter.__next__())]
                s = nodes[int(line_iter.__next__())]
                log_theta = np.float64(line_iter.__next__())
                element = (p,s)
                elements.append(element)
                theta[element] = np.exp(log_theta)
            left_vtree = p.vtree
            right_vtree = s.vtree
            assert p.vtree.parent == s.vtree.parent
            vtree = p.vtree.parent
            node = PSddNode(SddNode.DECOMPOSITION,elements,vtree,pmanager)
            node.theta = theta
            node.theta_sum = sum( theta.values() )
        if node:
            nodes[node_id] = node
    f.close()
    return node
