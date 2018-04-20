from hierarchical_variational_auto_encoder import HierarchicalVariationalAutoEncoder

vocab_size=1040
drop_prob = None
use_context_enhanced_rnn = True
use_pretrained_weights = True
hvae = HierarchicalVariationalAutoEncoder(vocab_size=vocab_size, drop_prob=drop_prob, \
		use_context_enhanced_rnn=use_context_enhanced_rnn, use_pretrained_weights=use_pretrained_weights, \
		max_rows=None, max_sentences_in_paragraph_loading=2e5, max_paragraphs=2e5)

# hvae.train_vae()

# hvae.train_guide()

# print('\nGENERATE')
# for sentence in hvae.generate_sentence():
#        print(sentence)

# print('\nINTERPOLATE')
# for sentence in hvae.interpolate("`` i want to talk to you '' .", "she did n't want to be with him ."):
# 	print(sentence)

# for sentence in hvae.interpolate("he was silent for a long moment .", "it was my turn ."):
# 	print(sentence)

# print('\nRECONSTRUCT')
# print(hvae.reconstruct("the man is very angry ."))
# print(hvae.reconstruct("`` hello, old friend '' ."))
# print(hvae.reconstruct("`` he is not coming tonight '' ."))
# print(hvae.reconstruct("it was a long day for many people ."))
# print(hvae.reconstruct("the house is on fire ."))
# print(hvae.reconstruct("we looked out at the sun ."))
# print(hvae.reconstruct("i went to the kitchen ."))
# print(hvae.reconstruct("how are you doing ? ."))
