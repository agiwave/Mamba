import aka.numpy as np

class TextStreamingLoader:
    def __init__(self, dataset, tokenizer, n_tokens, batch_size, data_mapper=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.n_tokens = n_tokens
        self.batch_size = batch_size
        self.data_mapper = data_mapper

        # Guess the number of batchs here.
        n_batchs = 0
        if len(dataset)>0:
            # first_item = self.__getitem_in_dataset__(dataset,0)
            n_batchs = dataset.dataset_size // 5 // batch_size // n_tokens
        self.n_batchs = n_batchs

    def __getitem_in_dataset__(self, dset, i):
        item = dset[i]
        if self.data_mapper:
            item = self.data_mapper(item)
        return item

    def __len__(self):
        return self.n_batchs

    def __iter__(self):
        class Args():
            def __init__(self, **kwargs): 
                for key in kwargs: setattr(self, key, kwargs[key])

        dset = self.dataset
        ntokens = self.n_tokens
        tokenizer = self.tokenizer
        nitems = len(dset)
        streams = [Args(
            datas = [],
            idx = i * (nitems//self.batch_size)
        ) for i in range(self.batch_size)]

        linecode = tokenizer.encode("\r\n")
        moveon = True
        while moveon:
            for stream in streams:
                while len(stream.datas) <= ntokens:
                    item = self.__getitem_in_dataset__(dset, stream.idx)
                    stream.idx = stream.idx + 1
                    if stream.idx == nitems:
                        moveon = False
                        break
                    stream.datas = stream.datas + linecode + tokenizer.encode(item)
                if not moveon:
                    break
                
            if moveon:
                inputs = [stream.datas[:ntokens] for stream in streams]
                targets = [stream.datas[1:ntokens+1] for stream in streams]
                for stream in streams:
                    stream.datas = stream.datas[ntokens:]
                yield np.array(inputs), np.array(targets)


        

        
        
