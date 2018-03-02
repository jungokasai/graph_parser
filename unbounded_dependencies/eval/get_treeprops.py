""" read in d6.treeproperties as a dictionary (t2props)
    read in d6.clean2.f.str as a dictionary (t2topsub)"""


""" Formats:
    
    In given txt file:
    t4240 root:S dir:RIGHT modif:AP lfronts:NP#s#0 rfronts:S#s#X ladjnodes:AP_S_VP_NP_N radjnodes:N_NP_VP_S_AP substnodes:S predaux:FALSE coanc:FALSE particle:FALSE particleShift:NA comp:n pred:TRUE dsubcat:NP#0 dsubcat2:NP#0 datshift:NA esubj:NO rel:NO wh:NO voice:NA


    Enter into dictionary as:
    t2props_dict[4240][root] == S
    etc.
    (where the treeid is an int)

"""






# read and format the properties for each tree,
def get_t2props_dict(fn):

    with open(fn, "r") as f:
        properties_txt = f.read()

    properties = properties_txt.split('\n')

    t2props_dict = {}

    for tree_props in properties:
        tree_props = tree_props.split(' ')

        if tree_props[0] == '':
            continue
        # use treeid as parent key (the integer part of e.g. t1966)
        tree_id = int(tree_props[0][1:])
        t2props_dict[tree_id] = {}

        # use property name as child keys for given treeid
        for prop in tree_props[1:]:
            prop = prop.split(':')
            prop_name = prop[0]
            prop_value = prop[1]
            t2props_dict[tree_id][prop_name] = prop_value
        
    return(t2props_dict)


# run the function
# t2props_dict = get_t2props_dict()



""" read in d6.clean2.f.str as a dictionary"""

""" Formats:
    
    In given txt file:
    t1966 NP##1#l# NP##2#l#f NP##2#r#f S##3#l# PP#2#4#l#s PP#2#4#r#s S##5#l# NP#0#6#l# -NONE-##7#l# -NONE-##7#r# NP#0#6#r# VP##8#l# V##9#l#h V##9#r#h NP#1#10#l#s NP#1#10#r#s PP##11#l# -NONE-##12#l# -NONE-##12#r# PP##11#r# VP##8#r# S##5#r# S##3#r# NP##1#r# 


    Enter into dictionary as:
    t2relation_dict[1966] == 0
    etc.
    (where the treeid is an int)

"""

# read and format the properties for each tree,
def get_t2topsub_dict(fn):
    t2topsub_dict = {}
    
    with open(fn, "r") as f:
        trees_txt = f.read()

    for tree_txt in trees_txt.split('\n'):    
        if tree_txt == '':
            continue
            
        for i, node in enumerate(tree_txt.split()):
            # if first node, store treeid
            if i == 0:
                treeid = int(node[1:])
            
            else:
                elements = node.split('#')
                if ((elements[0] == "NP") and 
                    (elements[1] in ["0", "1", "2", "3"])):
                    t2topsub_dict[treeid] = elements[1]
    
    return(t2topsub_dict)

if __name__ == '__main__':
    import os
    treeprops_file = "d6.treeproperties"
    get_t2props_dict(treeprops_file)
