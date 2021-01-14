Реализация алгоритма [LambdaRank](https://www.microsoft.com/en-us/research/uploads/prod/2016/02/MSR-TR-2010-82.pdf) на TensorFlow

Ноутбуки:
1. [notebooks/build_model.ipynb](https://github.com/naidenovaleksei/LambdaRank-tf/blob/master/notebooks/build_model.ipynb) - полный пайплайн построения модели и прогнозирование на тестовой выборке
2. [notebooks/train_tf.ipynb](https://github.com/naidenovaleksei/LambdaRank-tf/blob/master/notebooks/train_tf.ipynb) - вывод и проверка самой сложной части алгоритма - градиента ошибки LambdaRank
3. [notebooks/losses.ipynb](https://github.com/naidenovaleksei/LambdaRank-tf/blob/master/notebooks/losses.ipynb) - сравнение различных функций ошибки ранжирования (RankNet и LambdaRank)
4. [notebooks/learning_curve.ipynb](https://github.com/naidenovaleksei/LambdaRank-tf/blob/master/notebooks/learning_curve.ipynb) - визуализация кривых обучения для различных гиперпарамтров
