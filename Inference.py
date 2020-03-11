import numpy as np

class categories():
    def __init__(self, n_cat, n_cont, categorical, continuous):
        # set the number of variables of each type
        self.n_cat = n_cat
        self.n_cont = n_cont
        # categorical and continuous are lists of dictionaries
        # dictionaries for each category: subcategory -> label
        self.categorical = [categorical[i] for i in range(n_cat)] 
        
        # stocks the negative samples and their "not valued" variables
        # list of dictionaries, each sample is a dictionary
        self.neg_samples = set()
        
        # total number of samples received
        self.n_samples = 0

    def receive_sample(self, new_cat, label, verbose=True):
        """ we receive a judgment of a element
            input for the categorical variables is just the category, in a given order // list
            and for the continuous variables the input is their values, in a given order // list
            label is True if and only if the user approved the element shown
            EVERY CATEGORY MUST HAVE A INPUT VALUE
        """
        # min, max = 0, 0

        self.n_samples += 1
        if label is True:
            # receber as categorias na mesma ordem original (da classe)
            for i, cat in enumerate(new_cat):
                self.categorical[i][cat] = True
            
        
        if label is False:
            # there_is_false: verify if we know one of the categories values is known and it is negative
            # not_known_indexes: saves where are the categories values we dont know in this sample
            not_known_indexes = [] #0
            there_is_false = False
            for i, cat in enumerate(new_cat):
                if self.categorical[i][cat] is None:
                    not_known_indexes.append(i)
                if self.categorical[i][cat] is False:
                    there_is_false = True
                    return

            if len(not_known_indexes) == 0:
                #do nothing
                if verbose:
                    print ("---------------------actualized values---------------------")
                    print ("new sample received: " + str(new_cat))
                    print (self.categorical)
                    print (self.neg_samples)
                return 
                
            if (len(not_known_indexes) == 1):
                #infer as negative
                cat_index = not_known_indexes[0]
                cat_value = new_cat[cat_index]
                self.categorical[cat_index][cat_value] = False
                if verbose:
                    print ("---------------------actualized values---------------------")
                    print ("new sample received: " + str(new_cat))
                    print (self.categorical)
                    print (self.neg_samples)
                return

            if (len(not_known_indexes) > 1):
                #save the sample, conclusions can be done later.
                self.neg_samples.add(tuple(new_cat))
                if verbose:
                    print ("---------------------actualized values---------------------")
                    print ("new sample received: " + str(new_cat))
                    print (self.categorical)
                    print (self.neg_samples)
                return
            
        if verbose:
            print ("---------------------actualized values---------------------")
            print ("new sample received: " + str(new_cat))
            print (self.categorical)
            print (self.neg_samples)
            
            
    def actualize(self, verbose=True):
        if not self.neg_samples:
            return 
        
        run_remove_conclusionless_samples = False
        samples_to_remove = []
        for sample in self.neg_samples:
            not_known_indexes = []
            there_is_false = False
            for i, cat in enumerate(sample):
                if self.categorical[i][cat] is None:
                    not_known_indexes.append(i)
                    
                if self.categorical[i][cat] is False:
                    there_is_false = True
                    samples_to_remove.append(sample)
#                     self.neg_samples.remove(sample)
                    break
                    
            if not there_is_false and len(not_known_indexes) == 0:
                # could have a contradiciton
                raise RuntimeError
                            
            if not there_is_false and len(not_known_indexes) == 1:
                # infer label and active flag for conclusionless samples' remotion
                cat_index = not_known_indexes[0]
                cat_value = sample[cat_index]
                self.categorical[cat_index][cat_value] = False
                run_remove_conclusionless_samples = True
#                 return
            
        for sample in samples_to_remove:
            self.neg_samples.remove(sample)
            
        if run_remove_conclusionless_samples:
            self.actualize(verbose=False)
        
        if verbose:
            print ("---------------------final values---------------------")
            print (self.categorical)
            print (self.neg_samples)           
            


                
def main():
    categorical = [{"small": None, "medium": None, "big": None},  {"red": None, "blue": None, "yellow": None}, {"2doors": None, "4doors": None}]
                
    c = categories(3, 0, categorical, [])

    ncat = ['medium', 'blue', '4doors']
    c.receive_sample(ncat, label=True, verbose=True)

    ncat = ['big', 'red', '4doors']
    c.receive_sample(ncat, label=True, verbose=True)

    ncat = ['small', 'yellow', '2doors']
    c.receive_sample(ncat, label=False, verbose=True)

    ncat = ['small', 'red', '4doors']
    c.receive_sample(ncat, label=False, verbose=True)
    
    ncat = ['medium', 'yellow', '2doors']
    c.receive_sample(ncat, label=False, verbose=True)

    ncat = ['medium', 'yellow', '4doors']
    c.receive_sample(ncat, label=True, verbose=True)
    
    c.actualize()    
    
if __name__ =='__main__':
    main()