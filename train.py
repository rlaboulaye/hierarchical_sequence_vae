from hierarchical_variational_auto_encoder import HierarchicalVariationalAutoEncoder

vocab_size=1000
use_context_enhanced_rnn = False
use_pretrained_weights = False
hvae = HierarchicalVariationalAutoEncoder(vocab_size=vocab_size, \
		use_context_enhanced_rnn=use_context_enhanced_rnn, use_pretrained_weights=use_pretrained_weights, \
		max_rows=None, max_sentences_in_paragraph_loading=2e5, max_paragraphs=2e5)
hvae.train_vae(2,2,1)