

def declare_transformer():
    """Объявляем векторизатор"""
    from nltk.corpus import stopwords
    from sklearn.feature_extraction.text import TfidfVectorizer
    import nltk
    nltk.download('stopwords')
    stop_russian = stopwords.words('russian')
    text_transformer = TfidfVectorizer(stop_words=stop_russian, ngram_range=(1, 1), lowercase=True, max_features=15000)
    return text_transformer


def get_cvset():
    """Загружает базу вакансий"""
    import pandas as pd
    import csv
    csv.field_size_limit(100000000)
    vacs = pd.read_csv('vacancy_morphed.csv')
    print('Вакансии загружены')
    return vacs


def cv_preparation(text_cv, text_transformer):
    """ Предобрабатывает текст резюме и подготавливает для расчётов"""
    if not text_cv:
        return {'Error': "Текст отсутствует"}
    import re
    import pymorphy2
    import pandas as pd
    cv_exp = text_cv
    cv_exp_cleaned = re.sub(r'[^А-Яа-яЁёa-zA-Z]+', ' ', str(cv_exp))

    def lemmatize(row):
        morph = pymorphy2.MorphAnalyzer()
        t = []
        text_l = row
        for word in text_l.split():
            if len(word) <= 3:
                continue
            p = morph.parse(word)[0]
            t.append(p.normal_form)
        return " ".join(t)
    cv_morphed = lemmatize(cv_exp_cleaned)
    cv_ready_tf_idf = pd.DataFrame([cv_morphed])
    cv_ready_tf_idf.columns = ['a']
    '''трансформируем текст'''
    text_cv = text_transformer.transform(cv_ready_tf_idf['a'])
    print(f'Prepared text data')
    return text_cv


def preparation_set_func(vacs, text_transformer):
    """Подготавливает сет вакансий"""

    text_vacs = text_transformer.fit_transform(vacs['vacdesc_morphed'].values.astype(str))
    print(f'Calculation... {vacs.vacdesc_morphed.count()} rows')
    return text_vacs


def make_recommendations(text_cv, text_vacs, vacs, rows, vacancy_age):
    """Делает рекомендации"""
    import pandas as pd
    from datetime import datetime
    from sklearn.metrics.pairwise import linear_kernel
    cosine_similarities = linear_kernel(text_cv, text_vacs).flatten()
    vacs['cos_similarities'] = cosine_similarities
    #добавляем столбец с возрастом вакансии, в днях
    today = datetime.today()
    vacs['vacdate_datetime'] = [pd.to_datetime(line) for line in vacs.get('vacdate')]
    vacs['vacancy_age'] = vacs['vacdate_datetime'].apply(lambda x: (today - x).days)
    # и отсортируем по убыванию (т.к. cos(0)=1)
    vacs = vacs[vacs['vacancy_age'] <= vacancy_age].sort_values(by=['cos_similarities'], ascending=[0])
    data = []
    for index, row in vacs[0:rows].iterrows():
        vactitle = row['vactitle']
        cos_sim = row['cos_similarities']
        vac_age = row['vacancy_age']
        data.append([vactitle, cos_sim, vac_age])
    #recs_dict = {'input_text': text}
    #for i in range(10):
    # recs_dict[f'rec_id_{i}'] = f'Recommendation {i}'
    recs_dict = pd.DataFrame(data, columns=['Рекомендуемая вакансия', 'Степень сходства', 'Возраст вакансии'])
    return recs_dict


def main_calculation_func(text, rows, vacancy_age, verbose=True):
    """ Запускает расчет рекомендаций и выводит топ"""
    vacs = get_cvset()
    text_transformer = declare_transformer()
    text_vacs = preparation_set_func(vacs, text_transformer)
    text_cv = cv_preparation(text, text_transformer)
    recs_dict = make_recommendations(text_cv, text_vacs, vacs, rows, vacancy_age)
    if verbose:
        print(recs_dict)
    return recs_dict


if __name__ == "__main__":
    text = input("Введите текст\n")
    rows = int(input("Введите требуемое количество строк отчёта, или нажмите enter\n") or 10)
    vacancy_age = int(input("Введите макс.возраст вакансии (не менее 120, из-за возраста датасета), или нажмите enter\n") or 30)
    main_calculation_func(text, rows, vacancy_age)
