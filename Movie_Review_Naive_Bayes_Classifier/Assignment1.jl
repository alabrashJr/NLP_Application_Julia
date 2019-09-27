using DataStructures: DefaultDict
using Printf

#function to preprocessing the text
function clean_text(text)
punctuation=r"[(\(\)\",.!?;)]"
text=replace(lowercase(text),
    r"<.*?>"=>" ")
text=replace(
    text,
    punctuation=>"")
 text
end

println("reading training data...")

#read train data
postive_doc=Dict()
negative_doc=Dict()

foreach(readdir("./aclImdb/train/pos/")) do f
    #println("\nObject: ", f)
    postive_doc[f]=clean_text(read(string("./aclImdb/train/pos/",f),String))
end

foreach(readdir("./aclImdb/train/neg/")) do f
    #println("\nObject: ", f)
    negative_doc[f]=clean_text(read(string("./aclImdb/train/neg/",f),String))
end

println("training....")
#count function initialization
count_dict_pos = DefaultDict(0)
count_dict_neg=DefaultDict(0)
count_by_word_pos(x)=count_dict_pos[x]+=1
count_by_word_neg(x)=count_dict_neg[x]+=1

#count the words in the documents
for x in split.(values(postive_doc))
    count_by_word_pos.(x)
end
for x in split.(values(negative_doc))
    count_by_word_neg.(x)
end

#make gloabl dict for all words from both classes and remove the words that have been repeated less than the threshold which is 10
global_dict=merge(+,count_dict_pos,count_dict_neg)
global_dict=Dict(k=>global_dict[k] for k in keys(global_dict) if global_dict[k]>10)

#make dict for each class from the global_dict
w_count_dict_pos=Dict(k=>count_dict_pos[k] for k in keys(global_dict))
w_count_dict_neg=Dict(k=>count_dict_neg[k] for k in keys(global_dict))

#make dict for unk words for each class (every word has been repeated less than threshold is unk)
unkown_keys=setdiff(collect(keys(count_dict_pos)), collect(keys(w_count_dict_pos)))
unk_pos=Dict(k=>get(count_dict_pos,k,0) for k in unkown_keys)
unk_neg=Dict(k=>get(count_dict_neg,k,0) for k in unkown_keys)

#Calculate the umber of classes and their frequency 
total_neg_word_number=sum(values(w_count_dict_neg))
total_pos_word_number=sum(values(w_count_dict_pos))
pos_class_ratio=length(postive_doc)/(length(postive_doc)+length(negative_doc))
neg_class_ratio=length(negative_doc)/(length(postive_doc)+length(negative_doc))

#Calculate the probability of each word, besides calculate the unk probability for each class
dict_pos_pro=Dict(k=>w_count_dict_pos[k]/total_pos_word_number for k in keys(w_count_dict_pos))
dict_pos_pro["<UNK>"]=sum(values(unk_pos))/total_pos_word_number
dict_neg_pro=Dict(k=>w_count_dict_neg[k]/total_neg_word_number for k in keys(w_count_dict_neg))
dict_neg_pro["<UNK>"]=sum(values(unk_neg))/total_neg_word_number

#Generic functions to calcuate maxiumum likelyhood of Naive Bayes for a text in each class
get_prob_post_word(x)=get(dict_pos_pro,x,dict_pos_pro["<UNK>"])
get_prob_negat_word(x)=get(dict_neg_pro,x,dict_neg_pro["<UNK>"])
get_prob_post_text(x)=  log10(pos_class_ratio) + sum(log10.(get_prob_post_word.(split(clean_text(x)))))
get_prob_negat_text(x)=log10(neg_class_ratio) + sum(log10.(get_prob_negat_word.(split(clean_text(x)))))

let 
#initialze confuison matrix variables,and it's measure functions 
TP=0.0
FN=0.0
FP=0.0
TN=0.0

accuarcy(TP,FN,FP,TN)=(TN+TP)/(TP+FN+FP+TN)
recall(TP,FN)=TP/(TP+FN)
precesion(TP,FP)=TP/(TP+FP)

println("calculating the accuarcy...")
#Accuarcy calculating for Test data
for x in readdir("./aclImdb/train/pos/")
    #println("\nObject: ", f)
    text=read(string("./aclImdb/train/pos/",x),String)
    if get_prob_post_text(text) >=get_prob_negat_text(text); TP+=+1
    else;FP+= 1;end
end

for x in readdir("./aclImdb/test/neg/")
    text=clean_text(read(string("./aclImdb/test/neg/",x),String))
    if get_prob_negat_text(text)>=get_prob_post_text(text);TN+=1
    else;FN+=1;end
end




println("The accuarcy is $(@sprintf("%.2f", accuarcy(TP,FN,FP,TN)*100)) %")

println("The recall is $(@sprintf("%.2f", recall(TP,FN)*100)) %")

println("The precesion is $(@sprintf("%.2f", precesion(TP,FP)*100)) %")

println("\nTotal=$(TP+FN+FP+TN)\n\t\t\tActual True\tActual False\n\nTrue Predicition\t$TP\t\t$FP\n\nFalse Prediction\t$FN\t\t$TN")

while true
println("\n\nTry it on,write <stop> to kill the process")
t=readline()
if t=="<stop>";break;end
t=clean_text(t)
if get_prob_negat_text(t)>=get_prob_post_text(t);println("negative") ;else;println("postive") ; end
end

end