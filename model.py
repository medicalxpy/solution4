import datetime
import multiprocessing as mp
import os
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
import wordcloud
from scipy.special import softmax
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.early_stopping import EarlyStopping
from decoding_network import DecoderNetwork
from torch.nn import functional as F

class CTM:
    def __init__(
        self,
        bow_size,
        contextual_size,
        n_components=10,
        model_type="prodLDA",
        train_type="first",
        hidden_sizes=(100, 100),
        activation="softplus",
        dropout=0.2,
        learn_priors=True,
        batch_size=64,
        lr=2e-3,
        momentum=0.99,
        solver="adam",
        num_epochs=100,
        reduce_on_plateau=False,
        num_data_loader_workers=mp.cpu_count(),
        label_size=0,
        loss_weights=None,
        device="cuda:1",
        prior_mean=0.0,
        prior_variance=None
    ):
        self.prior_variance=prior_variance
        self.prior_mean=prior_mean
        self.train_type=train_type
        self.device =device
        self.bow_size = bow_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.contextual_size = contextual_size
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers
        self.training_doc_topic_distributions = None
        self.label_size = label_size
        if loss_weights:
            self.weights = loss_weights
        else:
            self.weights = {"beta": 1}

        self.model = DecoderNetwork(
            bow_size,
            self.contextual_size,
            n_components,
            model_type,
            hidden_sizes,
            activation,
            dropout,
            learn_priors,
            label_size=label_size,
            topic_prior_mean=self.prior_mean,
            topic_prior_variance=self.prior_variance
        )
        

        self.early_stopping = None

        # init optimizer
        if self.solver == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99)
            )
        elif self.solver == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum
            )

        # init lr scheduler

        self.scheduler = ReduceLROnPlateau(self.optimizer, 
                                           mode="min",
                                           factor=0.1,
                                           threshold=1e-4,
                                           patience=3)

        # performance attributes
        self.best_loss_train = float("inf")

        # training attributes
        self.model_dir = None
        self.nn_epoch = None

        # validation attributes
        self.validation_data = None

        # learned topics
        self.best_components = None

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

        self.model = self.model.to(self.device)

    def _loss(
        self,
        inputs,
        word_dists,
        prior_mean,
        prior_variance,
        posterior_mean,
        posterior_variance,
        posterior_log_variance,
    ):

        # KL term
        # var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        # diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum((diff_means * diff_means) / prior_variance, dim=1)
        # logvar det division term
        logvar_det_division = prior_variance.log().sum() - posterior_log_variance.sum(
            dim=1
        )
        # combine terms
        KL = 0.5 * (var_division + diff_term - self.n_components + logvar_det_division)

        # Reconstruction term
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)

        # loss = self.weights["beta"]*KL + RL

        return KL, RL
    

    # def _loss_TR(
    #         self,

    # ):
        
        
    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_samples in loader:
            # batch_size x vocab_size
            X_bow = batch_samples["X_bow"]
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_samples["X_contextual"]

            if "labels" in batch_samples.keys():
                labels = batch_samples["labels"]
                labels = labels.reshape(labels.shape[0], -1)
                labels = labels.to(self.device)
            else:
                labels = None

            if self.USE_CUDA:
                X_bow = X_bow.to(self.device)
                X_contextual = X_contextual.to(self.device)

            # forward pass
            self.model.zero_grad()
            (
                prior_mean,
                prior_variance,
                posterior_mean,
                posterior_variance,
                posterior_log_variance,
                word_dists,
                estimated_labels,
            ) = self.model(X_bow, X_contextual, labels)

            # backward pass
            kl_loss, rl_loss = self._loss(
                X_bow,
                word_dists,
                prior_mean,
                prior_variance,
                posterior_mean,
                posterior_variance,
                posterior_log_variance,
            )

            loss = self.weights["beta"] * kl_loss + rl_loss
            loss = loss.sum()

            if labels is not None:
                target_labels = torch.argmax(labels, 1)

                label_loss = torch.nn.CrossEntropyLoss()(
                    estimated_labels, target_labels
                )
                loss += label_loss

            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X_bow.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss,prior_mean,prior_variance

    def fit(
        self,
        train_dataset,
        validation_dataset=None,
        save_dir=None,
        verbose=False,
        early_stop=True,
        patience=5,
        delta=0,
        n_samples=20,
        do_train_predictions=True,
        return_mean = True,
        scheduler=True
    ):
        # Print settings to output file

        print(
            "Settings: \n\
                N Components: {}\n\
                Topic Prior Mean: {}\n\
                Topic Prior Variance: {}\n\
                Model Type: {}\n\
                Hidden Sizes: {}\n\
                Activation: {}\n\
                Dropout: {}\n\
                Learn Priors: {}\n\
                Learning Rate: {}\n\
                Momentum: {}\n\
                Reduce On Plateau: {}\n\
                Save Dir: {}".format(
                self.n_components,
                self.prior_mean,
                self.prior_variance,
                self.model_type,
                self.hidden_sizes,
                self.activation,
                self.dropout,
                self.learn_priors,
                self.lr,
                self.momentum,
                self.reduce_on_plateau,
                save_dir,
            )
        )

        self.model_dir = save_dir
        self.idx2token = train_dataset.idx2token
        train_data = train_dataset
        self.validation_data = validation_dataset
        if self.validation_data is not None:
            self.early_stopping = EarlyStopping(
                patience=patience, verbose=verbose, path=save_dir, delta=delta
            )
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_data_loader_workers,
            drop_last=True,
        )

        # init training variables
        samples_processed = 0
        best_loss = 10000000
        t = 0
        # train loop
        pbar = tqdm(self.num_epochs, position=0, leave=True)
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss,prior_mean,prior_variance= self._train_epoch(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()
            if verbose: 
                pbar.update(1)
            if self.validation_data is not None:
                validation_loader = DataLoader(
                    self.validation_data,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_data_loader_workers,
                    drop_last=True,
                )
                # train epoch
                s = datetime.datetime.now()
                val_samples_processed, val_loss = self._validation(validation_loader)
                e = datetime.datetime.now()

                # report
                if verbose:
                    print(
                        "Epoch: [{}/{}]\tSamples: [{}/{}]\tValidation Loss: {}\tTime: {}".format(
                            epoch + 1,
                            self.num_epochs,
                            val_samples_processed,
                            len(self.validation_data) * self.num_epochs,
                            val_loss,
                            e - s,
                        )
                    )


                self.early_stopping(val_loss, self)
                if self.early_stopping.early_stop:
                    print("Early stopping")

                    break
            else:
                # save last epoch
                self.best_components = self.model.beta
                if save_dir is not None:
                    self.save(save_dir)
            
            if early_stop:
                if train_loss < best_loss:
                    best_loss= train_loss
                    t=0
                else:
                    t = t+1
            if verbose:          
                pbar.set_description(
                    "Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                        epoch + 1,
                        self.num_epochs,
                        samples_processed,
                        len(train_data) * self.num_epochs,
                        train_loss,
                        e - s,
                    )
                )
            else:
                print("epoch:",epoch,"loss:",train_loss,"best_loss:",best_loss)
            if t > patience:
                print("early stop at epoch:",epoch+1)
                break
            if scheduler:
                self.scheduler.step(train_loss)
        
        pbar.close()
        if do_train_predictions:
            self.training_doc_topic_distributions = self.get_doc_topic_distribution(
                train_dataset, n_samples
            )


        if return_mean:
            post_mean=prior_mean.mean().item()
            post_variance=prior_variance.mean().item()
        else:

            post_mean=prior_mean
            post_variance=prior_variance   
                 
        return post_mean,post_variance

    def _validation(self, loader):
        """Validation epoch."""
        self.model.eval()
        val_loss = 0
        samples_processed = 0
        for batch_samples in loader:
            # batch_size x vocab_size
            X_bow = batch_samples["X_bow"]
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_samples["X_contextual"]

            if "labels" in batch_samples.keys():
                labels = batch_samples["labels"]
                labels = labels.to(self.device)
                labels = labels.reshape(labels.shape[0], -1)
            else:
                labels = None

            if self.USE_CUDA:
                X_bow = X_bow.to(self.device)
                X_contextual = X_contextual.to(self.device)

            # forward pass
            self.model.zero_grad()
            (
                prior_mean,
                prior_variance,
                posterior_mean,
                posterior_variance,
                posterior_log_variance,
                word_dists,
                estimated_labels,
            ) = self.model(X_bow, X_contextual, labels)

            kl_loss, rl_loss = self._loss(
                X_bow,
                word_dists,
                prior_mean,
                prior_variance,
                posterior_mean,
                posterior_variance,
                posterior_log_variance,
            )

            loss = self.weights["beta"] * kl_loss + rl_loss
            loss = loss.sum()

            if labels is not None:
                target_labels = torch.argmax(labels, 1)
                label_loss = torch.nn.CrossEntropyLoss()(
                    estimated_labels, target_labels
                )
                loss += label_loss

            # compute train loss
            samples_processed += X_bow.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss

    def get_thetas(self, dataset, n_samples=20):
        """
        Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via
        the parameter n_sample.

        :param dataset: a PyTorch Dataset containing the documents
        :param n_samples: the number of sample to collect to estimate the final distribution (the more the better).
        """
        return self.get_doc_topic_distribution(dataset, n_samples=n_samples)

    def get_doc_topic_distribution(self, dataset, n_samples=20, show_progress=False):
        """
        Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via
        the parameter n_sample.

        :param dataset: a PyTorch Dataset containing the documents
        :param n_samples: the number of sample to collect to estimate the final distribution (the more the better).
        :param show_progress: whether to show progress bar (default: True)
        """
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_data_loader_workers,
        )
        final_thetas = []
        with torch.no_grad():
            batch_iter = loader
            if show_progress:
                batch_iter = tqdm(loader, desc="Getting topic distributions")
            
            for batch_samples in batch_iter:
                # batch_size x vocab_size
                X_bow = batch_samples["X_bow"]
                X_bow = X_bow.reshape(X_bow.shape[0], -1)
                X_contextual = batch_samples["X_contextual"]

                if "labels" in batch_samples.keys():
                    labels = batch_samples["labels"]
                    labels = labels.to(self.device)
                    labels = labels.reshape(labels.shape[0], -1)
                else:
                    labels = None

                if self.USE_CUDA:
                    X_bow = X_bow.to(self.device)
                    X_contextual = X_contextual.to(self.device)

                # forward pass
                self.model.zero_grad()
                mu, log_var = self.model.get_posterior(X_bow, X_contextual, labels)
                thetas = self.model.sample(mu, log_var, n_samples).cpu().numpy()
                final_thetas.append(thetas)
        return np.concatenate(final_thetas, axis=0)

    def get_doc_topic_distribution_iterator(self, dataset, n_samples=20):
        """
        Get the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via
        the parameter n_sample. Returns an iterator over the document-topic distributions.

        :param dataset: a PyTorch Dataset containing the documents
        :param n_samples: the number of sample to collect to estimate the final distribution (the more the better).
        """
        self.model.eval()

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_data_loader_workers,
        )
        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                X_bow = batch_samples["X_bow"]
                X_bow = X_bow.reshape(X_bow.shape[0], -1)
                X_contextual = batch_samples["X_contextual"]

                if "labels" in batch_samples.keys():
                    labels = batch_samples["labels"]
                    labels = labels.to(self.device)
                    labels = labels.reshape(labels.shape[0], -1)
                else:
                    labels = None

                if self.USE_CUDA:
                    X_bow = X_bow.to(self.device)
                    X_contextual = X_contextual.to(self.device)

                # forward pass
                self.model.zero_grad()
                mu, log_var = self.model.get_posterior(X_bow, X_contextual, labels)
                thetas = self.model.sample(mu, log_var, n_samples).cpu().numpy()
                for theta in thetas:
                    yield theta

    def get_most_likely_topic(self, doc_topic_distribution):
        """get the most likely topic for each document

        :param doc_topic_distribution: ndarray representing the topic distribution of each document
        """
        return np.argmax(doc_topic_distribution, axis=0)

    def get_topics(self, k=10):
        """
        Retrieve topic words.

        :param k: int, number of words to return per topic, default 10.
        """
        assert k <= self.bow_size, "k must be <= input size."
        component_dists = self.best_components
        topics = defaultdict(list)
        for i in range(self.n_components):
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [self.idx2token[idx] for idx in idxs.cpu().numpy()]
            topics[i] = component_words
        return topics

    def get_topic_lists(self, k=10):
        """
        Retrieve the lists of topic words.

        :param k: (int) number of words to return per topic, default 10.
        """
        assert k <= self.bow_size, "k must be <= input size."
        # TODO: collapse this method with the one that just returns the topics
        component_dists = self.best_components
        topics = []
        for i in range(self.n_components):
            _, idxs = torch.topk(component_dists[i], k)
            component_words = [self.idx2token[idx] for idx in idxs.cpu().numpy()]
            topics.append(component_words)
        return topics

    def _format_file(self):
        model_dir = "contextualized_topic_model_nc_{}_tpm_{}_tpv_{}_hs_{}_ac_{}_do_{}_lr_{}_mo_{}_rp_{}".format(
            self.n_components,
            0.0,
            1 - (1.0 / self.n_components),
            self.model_type,
            self.hidden_sizes,
            self.activation,
            self.dropout,
            self.lr,
            self.momentum,
            self.reduce_on_plateau,
        )
        return model_dir

    def save(self, model_dir=None,part_name=None):
        """
        Save model. (Experimental Feature, not tested)

        :param models_dir: path to directory for saving NN models.
        """
        warnings.simplefilter("always", Warning)
        warnings.warn(
            "This is an experimental feature that we has not been fully tested. Refer to the following issue:"
            "https://github.com/MilaNLProc/contextualized-topic-models/issues/38",
            Warning,
        )

        if (self.model is not None) and (model_dir is not None):

            # model_dir = self._format_file()
            # if not os.path.isdir(os.path.join(models_dir, model_dir)):
            #     os.makedirs(os.path.join(models_dir, model_dir))

            filename = str(part_name)  + ".pth"
            fileloc = os.path.join(model_dir,filename)
            with open(fileloc, "wb") as file:
                torch.save(
                    {"state_dict": self.model.state_dict(), "dcue_dict": self.__dict__},
                    file,
                )

    def load(self, model_dir,part_name):
        """
        Load a previously trained model. (Experimental Feature, not tested)

        :param model_dir: directory where models are saved.
        :param epoch: epoch of model to load.
        """

        warnings.simplefilter("always", Warning)
        warnings.warn(
            "This is an experimental feature that we has not been fully tested. Refer to the following issue:"
            "https://github.com/MilaNLProc/contextualized-topic-models/issues/38",
            Warning,
        )

        epoch_file = str(part_name)+ ".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, "rb") as model_dict:
            checkpoint = torch.load(model_dict, map_location=torch.device(self.device))

        for (k, v) in checkpoint["dcue_dict"].items():
            setattr(self, k, v)

        self.model.load_state_dict(checkpoint["state_dict"])

    def get_topic_word_matrix(self):
        """
        Return the topic-word matrix (dimensions: number of topics x length of the vocabulary).
        If model_type is LDA, the matrix is normalized; otherwise the matrix is unnormalized.
        """
        topic_word_ma=F.softmax(self.model.topic_word_matrix,dim=1)
        return topic_word_ma.cpu().detach().numpy()
    

    def get_topic_word_distribution(self):
        """
        Return the topic-word distribution (dimensions: number of topics x length of the vocabulary).
        """
        mat = self.get_topic_word_matrix()
        return softmax(mat, axis=1)

    def get_word_distribution_by_topic_id(self, topic):
        """
        Return the word probability distribution of a topic sorted by probability.

        :param topic: id of the topic (int)

        :returns list of tuples (word, probability) sorted by the probability in descending order
        """
        if topic >= self.n_components:
            raise Exception("Topic id must be lower than the number of topics")
        else:
            wd = self.get_topic_word_distribution()
            t = [(word, wd[topic][idx]) for idx, word in self.idx2token.items()]
            t = sorted(t, key=lambda x: -x[1])
        return t

    def get_wordcloud(
        self, topic_id, n_words=5, background_color="black", width=1000, height=400
    ):
        """
        Plotting the wordcloud. It is an adapted version of the code found here:
        http://amueller.github.io/word_cloud/auto_examples/simple.html#sphx-glr-auto-examples-simple-py and
        here https://github.com/ddangelov/Top2Vec/blob/master/top2vec/Top2Vec.py

        :param topic_id: id of the topic
        :param n_words: number of words to show in word cloud
        :param background_color: color of the background
        :param width: width of the produced image
        :param height: height of the produced image
        """
        word_score_list = self.get_word_distribution_by_topic_id(topic_id)[:n_words]
        word_score_dict = {tup[0]: tup[1] for tup in word_score_list}
        plt.figure(figsize=(10, 4), dpi=200)
        plt.axis("off")
        plt.imshow(
            wordcloud.WordCloud(
                width=width, height=height, background_color=background_color
            ).generate_from_frequencies(word_score_dict)
        )
        plt.title("Displaying Topic " + str(topic_id), loc="center", fontsize=24)
        plt.show()

    def get_predicted_topics(self, dataset, n_samples):
        """
        Return the list containing the predicted topic for each document (length: number of documents).

        :param dataset: CTMDataset to infer topics
        :param n_samples: number of sampling of theta
        :return: the predicted topics
        """
        predicted_topics = []
        thetas = self.get_doc_topic_distribution(dataset, n_samples)

        for idd in range(len(dataset)):
            predicted_topic = np.argmax(thetas[idd] / np.sum(thetas[idd]))
            predicted_topics.append(predicted_topic)
        return predicted_topics

    def get_ldavis_data_format(self, vocab, dataset, n_samples):
        """
        Returns the data that can be used in input to pyldavis to plot
        the topics
        """
        term_frequency = np.ravel(dataset.X_bow.sum(axis=0))
        doc_lengths = np.ravel(dataset.X_bow.sum(axis=1))
        term_topic = self.get_topic_word_distribution()
        doc_topic_distribution = self.get_doc_topic_distribution(
            dataset, n_samples=n_samples
        )

        data = {
            "topic_term_dists": term_topic,
            "doc_topic_dists": doc_topic_distribution,
            "doc_lengths": doc_lengths,
            "vocab": vocab,
            "term_frequency": term_frequency,
        }

        return data

    def get_top_documents_per_topic_id(
        self, unpreprocessed_corpus, document_topic_distributions, topic_id, k=5
    ):
        probability_list = document_topic_distributions.T[topic_id]
        ind = probability_list.argsort()[-k:][::-1]
        res = []
        for i in ind:
            res.append(
                (unpreprocessed_corpus[i], document_topic_distributions[i][topic_id])
            )
        return res


    