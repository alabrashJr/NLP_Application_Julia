using Knet, Test, Base.Iterators, IterTools, Random # , LinearAlgebra, StatsBase
using AutoGrad: @gcheck  # to check gradients, use with Float64
Knet.atype() = KnetArray{Float32}  # determines what Knet.param() uses.
macro size(z, s); esc(:(@assert (size($z) == $s) string(summary($z),!=,$s))); end # for debugging

struct Vocab
    w2i::Dict{String,Int}
    i2w::Vector{String}
    unk::Int
    eos::Int
    tokenizer
end

function Vocab(file::String; tokenizer=split, vocabsize=Inf, mincount=1, unk="<unk>", eos="<s>")
    word_count = Dict{String,Int}()
    w2i = Dict{String,Int}()
    i2w = Vector{String}()
    int_unk = get!(w2i, unk, 1+length(w2i))
    int_eos = get!(w2i, eos, 1+length(w2i))
    for line in eachline(file)
        line = tokenizer(line)
        for word in line
            if haskey(word_count, word)
                word_count[word] += 1
            else
                word_count[word] = 1
            end
        end
    end
    word_count = collect(word_count)
    sort!(word_count, rev=true, by=x->x[2])
    # constructing w2i
    for pair in word_count
        if pair[2] >= mincount
            get!(w2i, pair[1], 1+length(w2i))
            if length(w2i) >= vocabsize
                break
            end
        end
    end
    w2i_array = collect(w2i)
    sort!(w2i_array, by=x->x[2])
    for pair in w2i_array
        push!(i2w, pair[1])
    end
    return Vocab(w2i, i2w, int_unk, int_eos, tokenizer)
end
struct TextReader
    file::String
    vocab::Vocab
end
function Base.iterate(r::TextReader, s=nothing)
    # Your code here
    s ==nothing ? file = open(r.file) : file =s
    
    if eof(file) == true
        close(file)
        return nothing
    end
    line = readline(file)
    text = r.vocab.tokenizer(line)
    arr = [get(r.vocab.w2i,word,r.vocab.w2i["<unk>"]) for word in text ]
    return (arr, file)
end

Base.IteratorSize(::Type{TextReader}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{TextReader}) = Base.HasEltype()
Base.eltype(::Type{TextReader}) = Vector{Int}

struct Embed; w; end

function Embed(vocabsize::Int, embedsize::Int)
    # Your code here
    Embed(param(embedsize, vocabsize, atype = KnetArray{Float32}))
end

function (l::Embed)(x)
    # Your code here
    l.w[:,x]
end

struct Linear; w; b; end

function Linear(inputsize::Int, outputsize::Int)
    # Your code here
    Linear(param(outputsize, inputsize, atype = KnetArray{Float32}), param0(outputsize, atype = KnetArray{Float32}))
end

function (l::Linear)(x)
    # Your code here
    l.w*x .+ l.b
end

function mask!(a,pad)
    # Your code here
    #b = deepcopy(a)
    for k in 1:size(a, 1)
        if a[k , size(a, 2)]!= pad
            continue
        end
        
        indices = []
        for i in 1:size(a[k, :], 1)
            if a[k, i] == pad
                push!(indices, i)
            end
        end
        indices = reverse(indices)
        for j in 1:size(indices, 1)-1
            if indices[j] == indices[j+1] + 1
                a[k, indices[j]] = 0
            else
                break
            end
        end
    end
    a
end
function Base.iterate(r::TextReader, s=nothing)
    # Your code here
    s ==nothing ? file = open(r.file) : file =s
    
    if eof(file) == true
        close(file)
        return nothing
    end
    line = readline(file)
    text = r.vocab.tokenizer(line)
    arr = [get(r.vocab.w2i,word,r.vocab.w2i["<unk>"]) for word in text ]
    return (arr, file)
end


datadir = "datasets/tr_to_en"

if !isdir(datadir)
    download("http://www.phontron.com/data/qi18naacl-dataset.tar.gz", "qi18naacl-dataset.tar.gz")
    run(`tar xzf qi18naacl-dataset.tar.gz`)
end

if !isdefined(Main, :tr_vocab)
    tr_vocab = Vocab("$datadir/tr.train", mincount=5)
    en_vocab = Vocab("$datadir/en.train", mincount=5)
    tr_train = TextReader("$datadir/tr.train", tr_vocab)
    en_train = TextReader("$datadir/en.train", en_vocab)
    tr_dev = TextReader("$datadir/tr.dev", tr_vocab)
    en_dev = TextReader("$datadir/en.dev", en_vocab)
    tr_test = TextReader("$datadir/tr.test", tr_vocab)
    en_test = TextReader("$datadir/en.test", en_vocab)
    @info "Testing data"
    @test length(tr_vocab.i2w) == 38126
    @test length(first(tr_test)) == 16
    @test length(collect(tr_test)) == 5029
end

struct MTData
    src::TextReader        # reader for source language data
    tgt::TextReader        # reader for target language data
    batchsize::Int         # desired batch size
    maxlength::Int         # skip if source sentence above maxlength
    batchmajor::Bool       # batch dims (B,T) if batchmajor=false (default) or (T,B) if true.
    bucketwidth::Int       # batch sentences with length within bucketwidth of each other
    buckets::Vector        # sentences collected in separate arrays called buckets for each length range
    batchmaker::Function   # function that turns a bucket into a batch.
end

function MTData(src::TextReader, tgt::TextReader; batchmaker = arraybatch, batchsize = 128, maxlength = typemax(Int),
                batchmajor = false, bucketwidth = 10, numbuckets = min(128, maxlength ÷ bucketwidth))
    buckets = [ [] for i in 1:numbuckets ] # buckets[i] is an array of sentence pairs with similar length
    MTData(src, tgt, batchsize, maxlength, batchmajor, bucketwidth, buckets, batchmaker)
end

Base.IteratorSize(::Type{MTData}) = Base.SizeUnknown()
Base.IteratorEltype(::Type{MTData}) = Base.HasEltype()
Base.eltype(::Type{MTData}) = NTuple{2}

function Base.iterate(d::MTData, state=nothing)
    # Your code here
    
    batch = nothing
    state = state
    if state == nothing
        for i in 1:length(d.buckets)
           d.buckets[i] = [] 
        end
    end
    
    while batch == nothing
        #print("W")
        if state == "index"
            #println("Hello. It's me.")
            found = false
            for i in 1:length(d.buckets)
                buck = d.buckets[i]
                if length(buck) != 0
                    #println(length(buck))
                    found = true
                    batch = d.batchmaker(d, buck) 
                    d.buckets[i] = []
                    break
                end
            end
            
            if found
                return (batch, "index") 
            else
                #print("Nothing")
                return nothing
            end
        end
        
        state_src, state_tgt = state==nothing ? (nothing, nothing) : state
        src_pair = iterate(d.src, state_src)
        tgt_pair = iterate(d.tgt, state_tgt)
        
        if src_pair == nothing
            #println("Index reached")
            state = "index"
            continue
        end

        src_sentence, state_src = src_pair
        tgt_sentence, state_tgt = tgt_pair
        
        if length(src_sentence) > d.maxlength
            state = (state_src, state_tgt)
            continue
        end
        if length(src_sentence) > length(d.buckets)*d.bucketwidth
            last_bucket = d.buckets[length(d.buckets)]
            push!(last_bucket, (src_sentence, tgt_sentence))
            if length(last_bucket) == d.batch_size
               batch = d.batchmaker(d, last_bucket)
               d.buckets[length(d.buckets)] = []
            end
        else 
            for i in 1:length(d.buckets)
               if (length(src_sentence) >= ((i-1)*d.bucketwidth+1)) & (length(src_sentence) <= (i*d.bucketwidth))
                   push!(d.buckets[i], (src_sentence, tgt_sentence))
                    if length(d.buckets[i]) == d.batchsize
                        #println("I am here")
                        batch = d.batchmaker(d, d.buckets[i])
                        d.buckets[i] = []
                    end
                    break
                end
            end   
        end
        state = (state_src, state_tgt)
    end
    #print("Bye")
    return (batch, state)
end

function arraybatch(d::MTData, bucket)
    # Your code here
    #println("Bucket: ", length(bucket), ", ", length(bucket[1][1]), ", ", length(bucket[1][2]))
    srclength = 0
    tgtlength = 0
    
    x = []
    y = []
    
    for pair in bucket
        src_sentence, tgt_sentence = pair
        if length(src_sentence) > srclength
           srclength = length(src_sentence)
        end
        if length(tgt_sentence) > tgtlength
           tgtlength = length(tgt_sentence)
        end
    end
    
    for pair in bucket
        src_sentence, tgt_sentence = pair
        src_sen = []
        tgt_sen = []
                
        #Target part. 
        tgt_eos = d.tgt.vocab.eos
        push!(tgt_sen, tgt_eos)
        for w in tgt_sentence
            push!(tgt_sen, w) 
        end
        while length(tgt_sen) != (tgtlength + 2)
            push!(tgt_sen, tgt_eos)
        end
        
        #Source part. 
        src_eos = d.src.vocab.eos
        eos_num = srclength - length(src_sentence)
        i = 0
        while i!= eos_num
            push!(src_sen, src_eos)
            i += 1
        end
        for w in src_sentence
            push!(src_sen, w) 
        end        
        push!(x, src_sen)
        push!(y, tgt_sen)
    end
    #println(size(hcat(x...)))
    #println(length(hcat(x...)))
    #println(length(hcat(x...)[1]))
    return Array(transpose(hcat(x...))), Array(transpose(hcat(y...)))
end

dtrn = MTData(tr_train, en_train)
ddev = MTData(tr_dev, en_dev)
dtst = MTData(tr_test, en_test)

x,y = first(dtst)

struct S2S_v1
    srcembed::Embed     # source language embedding
    encoder::RNN        # encoder RNN (can be bidirectional)
    tgtembed::Embed     # target language embedding
    decoder::RNN        # decoder RNN
    projection::Linear  # converts decoder output to vocab scores
    dropout::Real       # dropout probability to prevent overfitting
    srcvocab::Vocab     # source language vocabulary
    tgtvocab::Vocab     # target language vocabulary
end

function S2S_v1(hidden::Int,         # hidden size for both the encoder and decoder RNN
                srcembsz::Int,       # embedding size for source language
                tgtembsz::Int,       # embedding size for target language
                srcvocab::Vocab,     # vocabulary for source language
                tgtvocab::Vocab;     # vocabulary for target language
                layers=1,            # number of layers
                bidirectional=false, # whether encoder RNN is bidirectional
                dropout=0)           # dropout probability
    # Your code here
    src_embedd_layer = Embed(length(srcvocab.i2w), srcembsz)
    tgt_embedd_layer = Embed(length(tgtvocab.i2w), tgtembsz)
    proj = Linear(hidden, length(tgtvocab.i2w))
    encoder_layers = layers
    if bidirectional 
        encoder_layers /= 2
    end
    
    encoder = RNN(srcembsz, hidden, numLayers = encoder_layers, bidirectional = bidirectional, dropout = dropout)
    decoder = RNN(tgtembsz, hidden, numLayers = layers, dropout = dropout)
    S2S_v1(src_embedd_layer, encoder, tgt_embedd_layer, decoder, proj, dropout, srcvocab, tgtvocab)
end

function (s::S2S_v1)(src, tgt; average=true)
    # Your code here
#     @show size(src)
#     @show size(tgt)
    src_embed_out = s.srcembed(src) #;@show typeof(src_embed_out),size(src_embed_out)
    tgt_embed_out = s.tgtembed(tgt[:, 1:end-1]) #; @show size(tgt_embed_out)
    s.encoder.c, s.encoder.h = 0, 0
    y_en= s.encoder(src_embed_out) #;@show s.encoder
    s.decoder.h,s.decoder.c = s.encoder.c, s.encoder.h
    y_de = s.decoder(tgt_embed_out) #; @show size(y_de,1),size(y_de,2),size(y_de,3);
    y_reshaped = reshape(y_de, (size(y_de,1), size(y_de,2)*size(y_de,3))) # ;@show size(y_reshaped)
    scores = s.projection(y_reshaped) #;@show size(scores)
    dec_op=mask!(copy(tgt[:, 2:end]), s.srcvocab.eos)#;    @show size(dec_op)
    nll(scores,dec_op, average=average)
end

Knet.seed!(1)
model = S2S_v1(512, 512, 512, tr_vocab, en_vocab; layers=2, bidirectional=true, dropout=0.2)
(x,y) = first(dtst)

function loss(model, data; average=true)
    lss=Array{Float32,1}()
    if average
        istns=Array{Int,1}()
        for (x,y) in data
            ls,istn=model(x,y,average=false)
            push!(lss,ls)
            push!(istns,istn)
        end
        return sum(lss)/sum(istns)
    else
        istns=Int(0)
        for (x,y) in data; ls,t=model(x,y,average=average);push!(lss,ls);istns+=t;end
        return sum(lss),istns
    end
    
end

function train!(model, trn, dev, tst...)
    bestmodel, bestloss = deepcopy(model), loss(model, dev)
    progress!(adam(model, trn), steps=100) do y
        losses = [ loss(model, d) for d in (dev,tst...) ]
        if losses[1] < bestloss
            bestmodel, bestloss = deepcopy(model), losses[1]
        end
        return (losses...,)
    end
    return bestmodel
end

@info "Training S2S_v1"
epochs = 10
ctrn = collect(dtrn)
trnx10 = collect(flatten(shuffle!(ctrn) for i in 1:epochs))
trn20 = ctrn[1:20]
dev38 = collect(ddev)
#Uncomment this to train the model (This takes about 30 mins on a V100):
model = train!(model, trnx10, dev38, trn20)
#Uncomment this to save the model:
Knet.save("s2s_v1.1.jld2","model",model)
#Uncomment this to load the model:
#model = Knet.load("s2s_v1.jld2","model")

