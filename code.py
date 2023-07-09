
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
#from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix#, classification_report
import seaborn as sns
import statistics as st
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA

#%% Feature Selection & Extraction Parametreleri

feature_select = True
# Kaç değer seçilecek
selected_features = 8 # >= transformed_feature

feature_extract = True
# Kaç değere indirgenecek
transformed_features = 4 # > 1


#%% Veri Setinin Okunması

df = pd.read_csv('esk_rental_houses.csv') # PATH SHOULD BE GIVEN
df.head()

#%% Her öznitelik için özgün değerler

# =============================================================================
# print(df['province'].value_counts(), "\n")
# print(df['street'].value_counts(), "\n")
# print(df['size'].value_counts(), "\n")
# print(df['total_rooms'].value_counts(), "\n")
# print(df['total_floors'].value_counts(), "\n")
# print(df['floor'].value_counts(), "\n")
# print(df['age'].value_counts(), "\n")
# print(df['heating'].value_counts(), "\n")
# print(df['facing'].value_counts(), "\n")
# print(df['balcony'].value_counts(), "\n")
# print(df['furniture'].value_counts(), "\n")
# =============================================================================
print(df['rent'].value_counts(), "\n")
 
#%% String tipi değerleri kodlayarak kategorikleştirme

province_map = np.unique(df['province'])
streets_map = np.unique(df['street'])
heating_map = np.unique(df['heating'])

df['province']   = [np.where(province_map == x)[0][0] for x in df['province']]
df['street']   = [np.where(streets_map == x)[0][0] for x in df['street']]
df['heating']   = [np.where(heating_map == x)[0][0] for x in df['heating']]

#%% floor ve total_floors'tan daha anlamlı bir öznitelik oluşturulması

df.insert(2, 'floor_rated', [round(x['floor'] / x['total_floors'], 1) for x in df.iloc], True)

del df['floor']
del df['total_floors']
#df['floor_rated']

#%% Devamlı değerleri bucketing ile kategorikleştirme

size_interval = np.linspace(min(df['size']), max(df['size']), int((max(df['size'])-min(df['size']))/5))
age_interval = np.linspace(min(df['age']), max(df['age']), int((max(df['age'])-min(df['age']))/3))
rent_interval = np.linspace(min(df['rent']), max(df['rent']), int((max(df['rent'])-min(df['rent']))/200))

df['size'] = [int(x.mid) for x in pd.cut(df['size'], size_interval, include_lowest=True)]
df['age'] = [int(x.mid) for x in pd.cut(df['age'], age_interval, include_lowest=True)]
df['rent'] = [int(x.mid) for x in pd.cut(df['rent'], rent_interval, include_lowest=True)]

#%% Öznitelik - Target ayrımı

#df = df.sample(frac = 1)

target = 'rent'
labels = [f for f in df.columns]
labels.remove(target)

X = df[labels].values
y = df[target].values
X.shape


#%% Feature Selection 

x = SelectKBest(chi2, k = selected_features).fit_transform(X, y)

if feature_select:
    X = x.copy()

X.shape

#%% Feature Extraction, oluşan yeni değerler de bucketing yapılır

pca = PCA(n_components= transformed_features)
values = pca.fit_transform(X)
x = values

if feature_extract:
    X = x.copy()
    
    for col in [X[idx] for idx in range (len(X))]:
        col = [v.mid for v in pd.cut(col, np.linspace(min(col), max(col), 15), include_lowest=True)]
        
X.shape

#%% Test - Train ayrımı

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.1, random_state = 0)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#%% 10 Katlı Geçerleme iterasyonları boyunca her adımda bilgi toplayabilmek için oluşturulan diziler

estimators = [] # Tahminleyiciler
models = [] # fit edilmiş tahminleyiciler
predictions = [] # Tahminleyicilerin yaptığı tahminler
scores = [] # Tahminleyicilerin accuracy skorları
avg_scores = [] # Her fold için 5 modelin ortalama skoru
est_names = ['D.Tree(Gini)', 'D.Tree(Ent)', #'M.N.Bayes',
             'KNN(Dist)', 'KNN(Uni)', 'Lin.Reg.'] 
best_ests = [] # Her tahminleyici türü için 10 foldda görülmüş en iyi versiyonu tutar
y_tests = [] # Her folddaki test verisini ayrı ayrı tutar

estimators.append(DecisionTreeClassifier(criterion='gini', random_state=0)) 
estimators.append(DecisionTreeClassifier(criterion='entropy', random_state=0))
#estimators.append(MultinomialNB())
estimators.append(KNeighborsClassifier(weights = 'distance', algorithm = 'auto'))
estimators.append(KNeighborsClassifier(weights = 'uniform', algorithm = 'auto'))
estimators.append(LinearRegression())

#%% K-Fold

folds = 10
cv = KFold(n_splits = folds, random_state = 0, shuffle = False)

for train_index, test_index in cv.split(X):
    
    fold_models = []
    fold_scores = []
    fold_predictions = []
    
    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    y_tests.append(y_test.copy())
    
    for est in estimators:
        model = est.fit(X_train, y_train)
        
        target_predicted = model.predict(X_test)
        if (est == estimators[-1]): # Lineer Reg. tahminlerinin bucketing yapılması gerekiyor.
            target_predicted = [int(x.mid) for x in pd.cut(target_predicted, rent_interval, include_lowest=True)]
        
        score = model.score(X_test, y_test)
        
        fold_models.append(model)
        fold_scores.append(score)
        fold_predictions.append(target_predicted)
    
    models.append(fold_models)
    scores.append(fold_scores)
    predictions.append(fold_predictions)
    avg_scores.append(st.mean(fold_scores))
 
#%% Bu adımdan sonrası süreçle ilgili toplanan verilerin yansıtılması

print("\n")
for idx in range (len(estimators)):
    best_ests.append(np.max([x[idx] for x in scores]))
    print("In all folds, best", est_names[idx], "with accuracy:", round(best_ests[idx], 3), 
          "average:", round(st.mean([x[idx] for x in scores]), 3))

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(est_names,best_ests)
plt.title('Accuracy Values Amongst The Best Estimators In the All Folds Combined')
plt.show()  

#%%
    
best_fold = np.argmax(avg_scores)
print("\nBest fold was #", best_fold, " with an average score amongst the models of ", round(avg_scores[best_fold], 3))

print("\n")
drawn = False
for idx in range (len(estimators)):
    print("In the best fold,", est_names[idx], "with accuracy:", round(scores[best_fold][idx], 3))
    #print(classification_report(y_test, predictions[best_fold][idx]))
     
# =============================================================================
#     matrix = confusion_matrix(y_test, predictions[best_fold][idx])
#     print("Class Confusion Matrix\n", matrix)
#     if (scores[best_fold][idx] == max(scores[best_fold])) and (not drawn):
#         sns.heatmap(matrix, annot= True).set_ylim(9, 0)
#         drawn = True
# =============================================================================

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(est_names,scores[best_fold])
plt.title('Accuracy Values Amongst Different Estimators In the Best Fold')
plt.show()

#%%

best_estimator_of_all = np.argmax(best_ests)  
best_fold_for_best_estimator = np.argmax([x[best_estimator_of_all] for x in scores])
best_preds = predictions[best_fold_for_best_estimator][best_estimator_of_all]
best_test = y_tests[best_fold_for_best_estimator]
matrix = confusion_matrix(best_test, best_preds)

print("\nBest estimator was", est_names[best_estimator_of_all], "in fold #", best_fold_for_best_estimator,
      "with a score of", round(scores[best_fold_for_best_estimator][best_estimator_of_all], 3))

#print("\nClass Confusion Matrix\n")
sns.heatmap(matrix, annot= True).set_ylim(np.unique(best_test).size+2, 0)
