import torch
import torch.nn.functional as F
import pickle
from sklearn.preprocessing import normalize
from .utils import set_float32_highest_precision
import logging

@set_float32_highest_precision
class RankingSet(torch.nn.Module):
    def __init__(
        self,
        store=None,
        file_path=None,
        retrieve_path=None,
        idf_weights=None,
        debug=False,
        batch_size=0,
        CE_num_class=None,
        use_actaul_mw_for_retrival=None,
        need_to_normalize=True,
    ):
        """
        Creates a ranking set. Assumes the file specified at file_path is a pickle file of
        a numpy array of fingerprints. Fingerprints should be (n, 6144) dimension, and on load,
        the fingerprints will be normalized horizontally.

        store: whether to use an existing tensor (pass in), or None to signify loading from file

        file_path: file to load, if `store` is None

        idf_weights: optionally choose to perform a weighted ranking

        debug: enable/disable verbosity for debug statements
        """

        def normalize_if_needed(data, dim=1, p=2.0):
            if need_to_normalize:
                return F.normalize(data, dim=1, p=2.0)
            else:
                return data

        super().__init__()
        self.logger = logging.getLogger("lightning")
        self.logger.setLevel(logging.DEBUG)

        self.debug = debug
        self.idf = None
        self.batch_size = batch_size
        self.logger.debug("[Ranker] Initializing Ranker")

        with torch.no_grad():

            if use_actaul_mw_for_retrival is not None:
                import pickle
                with open(f'/root/gurusmart/MorganFP_prediction/inference_data/coconut_loutus_hyun_training/inference_metadata_latest_RDkit.pkl', 'rb') as file:
                    smiles_infos = pickle.load(file)
                    
                self.register_buffer(
                    "MWs",
                    torch.tensor([x[2] for x in smiles_infos]).float(),
                    persistent=False,
                )

            if store is not None:
                self.register_buffer(
                    "data", normalize_if_needed(store, dim=1, p=2.0), persistent=False
                )
            else:
                with open(file_path, "rb") as f:
                    rankingset_data = torch.load(f).type(torch.FloatTensor)
                    if CE_num_class is not None:
                        rankingset_data = torch.where(
                            rankingset_data >= CE_num_class,
                            CE_num_class - 1,
                            rankingset_data,
                        ).float()
                    self.register_buffer(
                        "data",
                        normalize_if_needed(rankingset_data, dim=1, p=2.0),
                        persistent=False,
                    )

            if retrieve_path:
                with open(retrieve_path, "rb") as f:
                    self.lookup = pickle.load(f)

            if idf_weights is not None:
                self.register_buffer("idf", idf_weights, persistent=False)
                self.register_buffer(
                    "idf_data",
                    normalize_if_needed(self.data * self.idf, dim=1, p=2.0),
                    persistent=False,
                )

        self.logger.info(f"[Ranking Set] Initialized with {len(self.data)} sample(s)")

    @staticmethod
    def round(fp):
        """
        Turns a fingerprint into a binary fingerprint, with 1's in indexes
        that have the max value

        fp: the fingerprint to convert
        """
        hi = torch.max(fp)
        where = torch.where(fp == hi)
        fp = torch.zeros(6144)
        fp[where] = 1
        return fp

    @staticmethod
    def normalized_to_nonzero(fp):
        hi = torch.max(fp)
        nonzero = torch.nonzero(torch.isclose(fp, hi))
        return tuple(nonzero[:, 0].tolist())

    # how many of the top n to be retrieved
    def retrieve_idx(self, query, n=50):
        query = F.normalize(query, dim=1, p=2.0).to(self.data.device).unsqueeze(0)
        # print(self.data.shape, query.shape)
        # (n x 6144) * (6144 x bs) = n x bs
        query_products = self.data @ query.T.squeeze(2)
        # query_products = (self.data @ query.T).flatten()
        # print(query_products.shape)
        _, idxs = torch.topk(query_products, k=n, dim=0)
        return idxs

    #   def retrieve(self, query, n=10):
    #     if not self.lookup:
    #       raise Exception("No retrieval dict")
    #     query = F.normalize(query, dim=0, p=2.0).to(self.device).unsqueeze(0)
    #     # (n x 6144) * (6144 x 1) = n x 1
    #     query_products = (self.data @ query.T).flatten()

    #     _, idxs = torch.topk(query_products, k=n)
    #     out = []
    #     for idx in idxs:
    #       nonzero = self.normalized_to_nonzero(self.data[idx])
    #       out.append(self.lookup.get(nonzero, None))
    #     return out

    def jaccard_rank(
        self,
        data,
        queries,
        truths,
        thresh,
        query_idx_in_rankingset,
        mw,
        use_actaul_mw_for_retrival,
    ):
        assert queries.size() == truths.size()
        if mw is not None:
            # filtering some mols in rankingset
            if use_actaul_mw_for_retrival:
                idx_to_keep = torch.abs(mw[0] - self.MWs) < 20
            else:
                idx_to_keep = torch.abs(self.MWs - mw[0]) / self.MWs < 0.2
            data = data[idx_to_keep]

        ct_greater = []
        with torch.no_grad():
            # compute jaccard of each query with each data
            for idx, query in enumerate(queries):
                jaccard = torch.sum((data * query) > 0, dim=1) / torch.sum(
                    (data + query) > 0, dim=1
                )
                ct_greater.append(torch.sum(jaccard >= thresh[idx], dtype=torch.int))
                # print("query", query.nonzero())
                # print("truths", truths[idx].nonzero())
                # print("data", data[0].nonzero())
                # # print("torch.sum(data*query, dim=1): ", torch.sum(data*query, dim=1))
                # # print("intersection: ", torch.sum(data*query, dim=1), torch.sum(data*query, dim=1).shape)
                # # print("union: ", torch.sum((data+query ) > 0, dim=1), torch.sum((data+query ) > 0, dim=1).shape)
                # print("jaccard: ", jaccard.sort(descending=True))
                # print("thresh: ", thresh, thresh.shape)
                # print(jaccard >= thresh[idx])
                # exit(0)
            ct_greater = torch.tensor(ct_greater)
            ct_greater -= (
                1  # subtract the label(the sample itself) from the ranking set
            )
        return ct_greater

    def dot_prod_rank(
        self,
        data,
        queries,
        truths,
        thresh,
        query_idx_in_rankingset,
        mw,
        use_actaul_mw_for_retrival,
    ):
        """
        Perform a dot-product ranking. Assumes that data, queries, and truths are already
        normalized.

        queries: (q, 6144) fingerprints to query (assumed normalized)

        truths: (q, 6144) of true fingerprints (assumed normalized)

        thresh: (1, q) what is the cutoff for things that ranked higher

        query_idx_in_rankingset: the first index of the same sample in the data tensors

        returns: size (q) tensor of how many were higher, minus how many labels were in the
          ranking set
        """
        assert queries.size() == truths.size()
        if mw is not None:
            # filtering some mols in rankingset
            if use_actaul_mw_for_retrival: # user has provided the actual mw
                idx_to_keep = torch.abs(mw[0] - self.MWs) < 20
            else:                           # 
                idx_to_keep = torch.abs(self.MWs - mw[0]) / self.MWs < 0.2
            data = data[idx_to_keep]

        with torch.no_grad():
            # q = queries.size()[0]
            # n = data.size()[0]
            # (n x 6144) * (6144 x q) = (n, q)

            # For all of these tensors, A_ij is the cosine similarity of sample n to query q
            query_products = data @ queries.T  # (n, q)

            # ((n, q) > (1, q)) -> (n, q) -> (1, q)

            # assert query_products.is_contiguous(), "query_products is not contiguous!"
            # assert thresh.is_contiguous(), "thresh is not contiguous!"
            # assert query_products.device == thresh.device, "Mismatched tensor devices!"
            # if  torch.isnan(query_products).any(): "NaN detected in query_products!"
            # if  torch.isnan(thresh).any(): "NaN detected in thresh!"
            # if  torch.isinf(query_products).any(): "Inf detected in query_products!"
            # if  torch.isinf(thresh).any(): "Inf detected in thresh!"
            # print(f"query_products shape: {query_products.shape}, device: {query_products.device}")
            # print(f"thresh shape: {thresh.shape}, device: {thresh.device}")

            ct_greater = torch.sum(
                torch.logical_or(
                    query_products >= thresh, torch.isclose(query_products, thresh)
                ),
                dim=0,
                keepdim=True,
                dtype=torch.int,
            )
            #   if (torch.sum((query_products >= thresh), dim=0) < torch.sum(torch.isclose(query_products, thresh), dim=0)).any():
            #     print(f"larger or equal to thresh: {torch.sum((query_products >= thresh), dim=0)}, is close to thresh: {torch.sum(torch.isclose(query_products, thresh), dim=0)}")
            #     print(torch.sum((query_products >= thresh), dim=0) < torch.sum(torch.isclose(query_products, thresh), dim=0))
            ct_greater = (
                ct_greater - 1
            )  # subtract the label(the sample itself) from the ranking set

            if self.debug:

                truth_products = data @ truths.T  # (n, q)
                print(truth_products.shape)
                print("truth_products: \n", truth_products)
                # inspect candidates larger than threshold
                # print("thresh: \n", thresh)
                # print("query_products: \n", query_products)
                # for i in range(q):
                #     print('biggest of current column: ', torch.topk(query_products[:,i], k=3).values)
                print("ct_greater: \n", ct_greater)
                # exit(0)
            return ct_greater

    def batched_rank(
        self,
        queries,
        truths,
        query_idx_in_rankingset,
        mw,
        use_actaul_mw_for_retrival,
        use_jaccard,
    ):
        """
        Perform a batched ranking

        queries: (q, 6144) predictions binary fingerprint tensor

        truths: (q, 6144) labels binary tensor

        returns: size (q) tensor of how many things in the ranking set had a higher cosine similarity
          cos(prediction, sample) than a threshold cos(prediction, label)
        """

        with torch.no_grad():
            if use_jaccard:
                intersection = torch.sum(queries * truths, dim=1)
                union = torch.sum((queries + truths) > 0, dim=1)
                thresh = intersection / union
                return self.jaccard_rank(
                    self.data,
                    queries,
                    truths,
                    thresh,
                    query_idx_in_rankingset,
                    mw,
                    use_actaul_mw_for_retrival,
                )
            else:
                #   q = queries.size()[0]
                queries = F.normalize(queries, dim=1, p=2.0)  # (q, 6144)
                truths = F.normalize(truths, dim=1, p=2.0)  # (q, 6144)

                # transpose and element-wise dot ->(6144, q)
                # sum dim 0, keepdims -> (1, q)
                thresh = torch.sum((queries * truths).T, dim=0, keepdim=True)

                return self.dot_prod_rank(
                    self.data,
                    queries,
                    truths,
                    thresh,
                    query_idx_in_rankingset,
                    mw,
                    use_actaul_mw_for_retrival,
                )

    def batched_rank_tfidf(self, queries, truths):
        """
        Perform a batched ranking with tfidf

        query(predictions) should be (q, 6144)
        truths should be (q, 6144)

        returns the number of items that have a higher dot product similarity (query_i (dot) truth_i)
        """
        if self.idf_data is None:
            raise Exception("No idf weights")
        with torch.no_grad():
            q = queries.size()[0]
            queries = F.normalize(queries.to(self.device) * self.idf, dim=1, p=2.0)
            truths = F.normalize(truths.to(self.device) * self.idf, dim=1, p=2.0)
            thresh = torch.stack(
                [torch.dot(queries[i, :], truths[i, :]) for i in range(q)]
            ).to(
                self.device
            )  # (q)

            return self.dot_prod_rank(self.idf_data, queries, truths, thresh)
