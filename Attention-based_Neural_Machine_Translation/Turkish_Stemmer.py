import re
from TurkishStemmer import TurkishStemmer
import argparse
stemmer = TurkishStemmer()



def stem(x):
    """this method takes Turkish word and return the list of word and it's suffixes"""
    lst=[]
    x=re.sub('[^A-Za-z0-9 üöçşığ]+',"",x).strip()
    while len(x)>0:
        stemmezied=stemmer.stem(x)
        x_post=re.sub(stemmezied,"",x)
        if x_post==x:
            x=re.sub(stemmezied[:-1],"",x)[1:]
        else:
            x=x_post
        lst.append(stemmezied)
    return lst


def get_args():
    '''
    This function parses and return arguments passed in
    '''
    parser = argparse.ArgumentParser(prog='Turkish Stemmerizer',
                                     description='takes input file path to write its stemmerizered version')
    parser.add_argument('--input_file', help="Input file")
    args = parser.parse_args()

    return(args)

if __name__ == "__main__":
    args = get_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        text=f.readlines()

    text_post=[]
    for i,x in enumerate(text):
        sentence_post=[]
        for w in x.split():
            try:
                st=" ".join(stem(w))
                sentence_post.append(st)

            except Exception:
                print(i,"first for,\n",x,"\n",w)
                raise RuntimeError
        if len(sentence_post)>1:text_post.append(" ".join(sentence_post))#;print(x,"=>\n"," ".join(sentence_post),"\n","".join(["="]*40))
        else:text_post.append("".join(sentence_post))#;print(x,"=>\n",sentence_post,"\n","".join(["="]*40))
    
    f_output=open(args.input_file.split("/")[-1]+".stemmerized","w")
    for sentence in text_post:
        f_output.write(sentence.strip())
        f_output.write('\n')
    f_output.close()