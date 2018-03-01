""" helper functions for debugging"""

def get_stags(sent, stag):
    sent = sent.split()    
    output = []
    for i in range(len(sent)):
        output.append((sent[i], stag[i]))
    return(output)


def get_graphviz_format(k, overlay_trees=False):

    """ gather our linguistic info """

    sent_h = sents_h[k]
    parse_h = parses_h[k]
    #pos_h = all_pos_h[k]

    stag_h = stags_h[k].split()
    sent_stag_h = ['\n'.join([sent, stag]) for sent, stag in get_stags(sent_h, stag_h)]


    sent_t = sents_t[k]
    parse_t = parses_t[k]
    #pos_t = all_pos_t[k]

    stag_t = stags_t[k].split()

    sent_stag_t = ['\n'.join([sent, stag]) for sent, stag in get_stags(sent_t, stag_t)]

#     if text_or_hypo == 't':
#         sent_stag = sent_stag_t
#     else:
#         sent_stag = sent_stag_h


#     for t_or_h in ["t", "h"]:
#         if t_or_h == "t":
#             sent_stag = sent_stag_t
#             sent = sent_t
#         else:
#             sent_stag = sent_stag_h
#             sent = sent_h

    lex_stag_parse_h = lexicalize(parse_h, sent_h, ['root'] + stag_h)

    lex_stag_parse_t = lexicalize(parse_t, sent_t, ['root'] + stag_t) 


    if not overlay_trees:

        """ for t """
        lex_stag_parse = lex_stag_parse_t

        output = ""

        output += "digraph t {\n"
        for w1, w2, dep in lex_stag_parse:
            output += "\"" + w2 + "\"" + " -> "+ "\"" + w1 + "\"" + " [ label = \"" + dep + "\" ]; " + "\n"

        output += "}"

    #     print(output)
        output_t = output

        print()
        print()


        """ for h """

        lex_stag_parse = lex_stag_parse_h

        output = ""

        output += "digraph h {\n"
        for w1, w2, dep in lex_stag_parse:
            output += "\"" + w2 + "\"" + " -> "+ "\"" + w1 + "\"" + " [ label = \"" + dep + "\"" + "  fontcolor=red" + "]; " + "\n"

        output += "}"

    #     print(output)

        output_h = output

        return(output_t + '\n\n' + output_h)
    
    
    # overlay_trees
    else:
        """ for t """
        lex_stag_parse = lex_stag_parse_t

        output = ""

        output += "digraph t_and_h {\n"
        for w1, w2, dep in lex_stag_parse:
            output += "\"" + w2 + "\"" + " -> "+ "\"" + w1 + "\"" + " [ label = \"" + dep + "\" ]; " + "\n"



        """ for h """
        output += "\n\n"

        lex_stag_parse = lex_stag_parse_h
        

        for w1, w2, dep in lex_stag_parse:
            output += "\"" + w2 + "\"" + " -> "+ "\"" + w1 + "\"" + " [ label = \"" + dep + "\"" + "  fontcolor=red" + "]; " + "\n"

        output += "}"

        return(output)
    


    """
    Use with 
    http://www.webgraphviz.com/

    Model:

    digraph G {
     a -> b [ label="a to b" ];
     b -> c [ label="another label"];
    }




    Or look into
    http://matthiaseisen.com/articles/graphviz/



    ADD: find the breaking case in entailments. 
    print it twice, or somehow print with different color. 
    
    ADD: print all of parse_h in different color. (use words only, not tree tags?)
    
    (With plotly:)
    WANT: hover, get tree_properties. 
    WANT: 

    """
