import torch
from torch import nn
from torch.nn import functional as F

from inference_network import  CombinedInferenceNetwork

class DecoderNetwork(nn.Module):

    def __init__(
        self,
        input_size,
        bert_size,
        n_components=10,
        model_type="prodLDA",
        hidden_sizes=(100, 100),
        activation="softplus",
        dropout=0.2,
        learn_priors=True,
        label_size=0,
        topic_prior_mean= 0.0,
        topic_prior_variance= None
    ):
        super(DecoderNetwork, self).__init__()
        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.topic_word_matrix = None


        self.inf_net = CombinedInferenceNetwork(
                input_size,
                bert_size,
                n_components,
                hidden_sizes,
                activation,
                label_size=label_size,
            )
        if label_size != 0:
            self.label_classification = nn.Linear(n_components, label_size)

        if isinstance(topic_prior_mean, torch.Tensor):
            self.prior_mean=topic_prior_mean
        else:
            self.prior_mean = torch.tensor([topic_prior_mean] * n_components)
        if torch.cuda.is_available():
            self.prior_mean = self.prior_mean.cuda()
        if self.learn_priors:
            self.prior_mean = nn.Parameter(self.prior_mean)

        if isinstance(topic_prior_variance, torch.Tensor):
            self.prior_variance=topic_prior_variance
        else:    
            topic_prior_variance = 1.0 - (1.0 / self.n_components)
            self.prior_variance = torch.tensor([topic_prior_variance] * n_components)
        if torch.cuda.is_available():
            self.prior_variance = self.prior_variance.cuda()
        if self.learn_priors:
            self.prior_variance = nn.Parameter(self.prior_variance)




        self.beta = torch.Tensor(n_components, input_size)
        if torch.cuda.is_available():
            self.beta = self.beta.cuda()
        self.beta = nn.Parameter(self.beta)
        nn.init.xavier_uniform_(self.beta)


        self.beta_batchnorm = nn.BatchNorm1d(input_size, affine=False)


        # dropout on theta
        self.drop_theta = nn.Dropout(p=self.dropout)

    @staticmethod
    def reparameterize(mu, logvar):
        """Reparameterize the theta distribution."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, x, x_bert, labels=None):
        """Forward pass."""
        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)
        posterior_sigma = torch.exp(posterior_log_sigma)

        # generate samples from theta
        theta = F.softmax(self.reparameterize(posterior_mu, posterior_log_sigma), dim=1)
        theta = self.drop_theta(theta)

        # prodLDA vs LDA
        if self.model_type == "prodLDA":
            # in: batch_size x input_size x n_components
            word_dist = F.softmax(
                self.beta_batchnorm(torch.matmul(theta, self.beta)), dim=1
            )
            # word_dist: batch_size x input_size
            self.topic_word_matrix = self.beta
        elif self.model_type == "LDA":
            # simplex constrain on Beta
            beta = F.softmax(self.beta_batchnorm(self.beta), dim=1)
            self.topic_word_matrix = beta
            word_dist = torch.matmul(theta, beta)
            # word_dist: batch_size x input_size
        else:
            raise NotImplementedError("Model Type Not Implemented")

        # classify labels

        estimated_labels = None

        if labels is not None:
            estimated_labels = self.label_classification(theta)

        return (
            self.prior_mean,
            self.prior_variance,
            posterior_mu,
            posterior_sigma,
            posterior_log_sigma,
            word_dist,
            estimated_labels,
        )

    def get_posterior(self, x, x_bert, labels=None):
        """Get posterior distribution."""
        # batch_size x n_components
        posterior_mu, posterior_log_sigma = self.inf_net(x, x_bert, labels)

        return posterior_mu, posterior_log_sigma

    def get_theta(self, x, x_bert, labels=None):
        with torch.no_grad():
            # batch_size x n_components
            posterior_mu, posterior_log_sigma = self.get_posterior(x, x_bert, labels)
            # posterior_sigma = torch.exp(posterior_log_sigma)

            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=1
            )

            return theta

    def sample(self, posterior_mu, posterior_log_sigma, n_samples: int = 20):
        with torch.no_grad():
            posterior_mu = posterior_mu.unsqueeze(0).repeat(n_samples, 1, 1)
            posterior_log_sigma = posterior_log_sigma.unsqueeze(0).repeat(n_samples, 1, 1)
            # generate samples from theta
            theta = F.softmax(
                self.reparameterize(posterior_mu, posterior_log_sigma), dim=-1
            )

            return theta.mean(dim=0)