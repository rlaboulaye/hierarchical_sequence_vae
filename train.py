from hierarchical_variational_auto_encoder import HierarchicalVariationalAutoEncoder

vocab_size=1040
drop_prob = None
use_context_enhanced_rnn = True
use_pretrained_weights = True
hvae = HierarchicalVariationalAutoEncoder(vocab_size=vocab_size, drop_prob=drop_prob, \
		use_context_enhanced_rnn=use_context_enhanced_rnn, use_pretrained_weights=use_pretrained_weights, \
		max_rows=5e3, max_sentences_in_paragraph_loading=5e3, max_paragraphs=5e3)

# hvae.train_vae()

# hvae.train_guide()

# print('\nGENERATE SENTENCE')
# for sentence in hvae.generate_sentence(35):
#        print(sentence)

# print('\nGENERATE PARAGRAPH')
# hvae.generate_paragraph("yes , you did .")
# print()
# hvae.generate_paragraph("he asked , knowing she was n't .")
# print()
# hvae.generate_paragraph("what are you two doing up here ? .")
# print()
# hvae.generate_paragraph("you need someone down by the door .")
# print()
# hvae.generate_paragraph("`` what did you do now ? '' .")
# print()
# hvae.generate_paragraph("i miss him so much .")
# print()

# print('\nINTERPOLATE')
# for sentence in hvae.interpolate("`` i want to talk to you '' .", "she did n't want to be with him ."):
# 	print(sentence)

# for sentence in hvae.interpolate("he was silent for a long moment .", "it was my turn ."):
# 	print(sentence)

# for sentence in hvae.interpolate("he looked at me .", "then he kissed me ."):
# 	print(sentence)

# for sentence in hvae.interpolate("he is my best friend .", "we had a big fight ."):
# 	print(sentence)

# for sentence in hvae.interpolate("yes , you did .", "are you serious ? ."):
# 	print(sentence)
# print()
# for sentence in hvae.interpolate("he asked , knowing she was n't .", "then he walked away ."):
# 	print(sentence)
# print()
# for sentence in hvae.interpolate("what are you two doing up here ? .", "its just that ! ."):
# 	print(sentence)
# print()
# for sentence in hvae.interpolate("you need someone down by the door .", "how did you find her ? ."):
# 	print(sentence)
# print()
# for sentence in hvae.interpolate("`` what did you do now ? '' .", "you could have done as well '' ."):
# 	print(sentence)
# print()

# print('\nRECONSTRUCT')
# print(hvae.reconstruct("the man is very angry ."))
# print(hvae.reconstruct("`` hello, old friend '' ."))
# print(hvae.reconstruct("`` he is not coming tonight '' ."))
# print(hvae.reconstruct("it was a long day for many people ."))
# print(hvae.reconstruct("the house is on fire ."))
# print(hvae.reconstruct("we looked out at the sun ."))
# print(hvae.reconstruct("i went to the kitchen ."))
# print(hvae.reconstruct("how are you doing ? ."))
