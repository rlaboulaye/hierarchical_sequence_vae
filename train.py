from hierarchical_variational_auto_encoder import HierarchicalVariationalAutoEncoder

vocab_size=1000
drop_prob = None
use_context_enhanced_rnn = True
use_pretrained_weights = False
hvae = HierarchicalVariationalAutoEncoder(vocab_size=vocab_size, drop_prob=drop_prob, \
		use_context_enhanced_rnn=use_context_enhanced_rnn, use_pretrained_weights=use_pretrained_weights, \
		max_rows=None, max_sentences_in_paragraph_loading=2e5, max_paragraphs=2e5)

hvae.train_vae()

# for sentence in hvae.generate_sentence():
#        print(sentence)

# for interpolation in hvae.interpolate():
# 	print()
# 	for sentence in interpolation:
# 		print(sentence)

# sentences, reconstructed_sentences = hvae.test_reconstruction()
# for pair in zip(sentences, reconstructed_sentences):
# 	print(pair)
