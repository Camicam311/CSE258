
## Regression

### Task 1


```python
def parseData(fname):
  for l in open(fname):
    yield eval(l)

print "Reading data..."
data = list(parseData("beer.json"))

counts = dict()
average = dict()

for data_point in data:
    style = data_point['beer/style']
    review = data_point['review/taste']
    n = counts.get(style, 0)
    old_average = average.get(style, 0)
    
    counts[style] = n + 1
    average[style] = (old_average * n + review) / (n+1)

print "\n#Reviews for each style"
counts

```

    Reading data...
    
    #Reviews for each style
    {'Altbier': 165,
     'American Adjunct Lager': 242,
     'American Amber / Red Ale': 665,
     'American Amber / Red Lager': 42,
     'American Barleywine': 825,
     'American Black Ale': 138,
     'American Blonde Ale': 357,
     'American Brown Ale': 314,
     'American Dark Wheat Ale': 14,
     'American Double / Imperial IPA': 3886,
     'American Double / Imperial Pilsner': 14,
     'American Double / Imperial Stout': 5964,
     'American IPA': 4113,
     'American Malt Liquor': 90,
     'American Pale Ale (APA)': 2288,
     'American Pale Lager': 123,
     'American Pale Wheat Ale': 154,
     'American Porter': 2230,
     'American Stout': 591,
     'American Strong Ale': 166,
     'American Wild Ale': 98,
     'Baltic Porter': 514,
     'Belgian Dark Ale': 175,
     'Belgian IPA': 128,
     'Belgian Pale Ale': 144,
     'Belgian Strong Dark Ale': 146,
     'Belgian Strong Pale Ale': 632,
     'Berliner Weissbier': 10,
     'Bi\xc3\xa8re de Garde': 7,
     'Black & Tan': 122,
     'Bock': 148,
     'Braggot': 26,
     'California Common / Steam Beer': 11,
     'Chile Beer': 11,
     'Cream Ale': 69,
     'Czech Pilsener': 1501,
     'Doppelbock': 873,
     'Dortmunder / Export Lager': 31,
     'Dubbel': 165,
     'Dunkelweizen': 61,
     'Eisbock': 8,
     'English Barleywine': 133,
     'English Bitter': 267,
     'English Brown Ale': 495,
     'English Dark Mild Ale': 21,
     'English India Pale Ale (IPA)': 175,
     'English Pale Ale': 1324,
     'English Pale Mild Ale': 21,
     'English Porter': 367,
     'English Stout': 136,
     'English Strong Ale': 164,
     'Euro Dark Lager': 144,
     'Euro Pale Lager': 701,
     'Euro Strong Lager': 329,
     'Extra Special / Strong Bitter (ESB)': 667,
     'Flanders Oud Bruin': 13,
     'Flanders Red Ale': 2,
     'Foreign / Export Stout': 55,
     'Fruit / Vegetable Beer': 1355,
     'German Pilsener': 586,
     'Hefeweizen': 618,
     'Herbed / Spiced Beer': 73,
     'Irish Dry Stout': 101,
     'Irish Red Ale': 83,
     'Keller Bier / Zwickel Bier': 23,
     'Kristalweizen': 7,
     'K\xc3\xb6lsch': 94,
     'Lambic - Fruit': 6,
     'Lambic - Unblended': 10,
     'Light Lager': 503,
     'Low Alcohol Beer': 7,
     'Maibock / Helles Bock': 225,
     'Milk / Sweet Stout': 69,
     'Munich Dunkel Lager': 141,
     'Munich Helles Lager': 650,
     'M\xc3\xa4rzen / Oktoberfest': 557,
     'Oatmeal Stout': 102,
     'Old Ale': 1052,
     'Pumpkin Ale': 560,
     'Quadrupel (Quad)': 119,
     'Rauchbier': 1938,
     'Russian Imperial Stout': 2695,
     'Rye Beer': 1798,
     'Saison / Farmhouse Ale': 141,
     'Schwarzbier': 53,
     'Scotch Ale / Wee Heavy': 2776,
     'Scottish Ale': 78,
     'Scottish Gruit / Ancient Herbed Ale': 65,
     'Smoked Beer': 61,
     'Tripel': 257,
     'Vienna Lager': 33,
     'Weizenbock': 13,
     'Wheatwine': 455,
     'Winter Warmer': 259,
     'Witbier': 162}




```python
print "\nAverage reviews for each style"
average
```

    
    Average reviews for each style





    {'Altbier': 3.4030303030303024,
     'American Adjunct Lager': 2.9483471074380154,
     'American Amber / Red Ale': 3.513533834586466,
     'American Amber / Red Lager': 3.6904761904761907,
     'American Barleywine': 4.064242424242424,
     'American Black Ale': 3.8731884057971016,
     'American Blonde Ale': 3.2549019607843137,
     'American Brown Ale': 3.7436305732484074,
     'American Dark Wheat Ale': 3.6785714285714284,
     'American Double / Imperial IPA': 4.033324755532658,
     'American Double / Imperial Pilsner': 3.8214285714285716,
     'American Double / Imperial Stout': 4.479963112005356,
     'American IPA': 4.000850960369554,
     'American Malt Liquor': 2.255555555555555,
     'American Pale Ale (APA)': 3.649694055944056,
     'American Pale Lager': 3.2154471544715446,
     'American Pale Wheat Ale': 3.3344155844155843,
     'American Porter': 4.081838565022416,
     'American Stout': 3.8197969543147208,
     'American Strong Ale': 3.569277108433735,
     'American Wild Ale': 4.188775510204083,
     'Baltic Porter': 4.213035019455248,
     'Belgian Dark Ale': 3.34,
     'Belgian IPA': 3.94921875,
     'Belgian Pale Ale': 3.7395833333333335,
     'Belgian Strong Dark Ale': 3.6952054794520546,
     'Belgian Strong Pale Ale': 4.05617088607595,
     'Berliner Weissbier': 3.55,
     'Bi\xc3\xa8re de Garde': 3.9285714285714284,
     'Black & Tan': 3.9426229508196724,
     'Bock': 3.1891891891891886,
     'Braggot': 3.8076923076923075,
     'California Common / Steam Beer': 3.3181818181818183,
     'Chile Beer': 3.9545454545454546,
     'Cream Ale': 3.028985507246377,
     'Czech Pilsener': 3.6095936042638246,
     'Doppelbock': 3.9828178694158076,
     'Dortmunder / Export Lager': 3.4193548387096775,
     'Dubbel': 3.7363636363636363,
     'Dunkelweizen': 3.4918032786885247,
     'Eisbock': 3.75,
     'English Barleywine': 4.360902255639096,
     'English Bitter': 3.5374531835205993,
     'English Brown Ale': 3.728282828282828,
     'English Dark Mild Ale': 3.7857142857142856,
     'English India Pale Ale (IPA)': 3.4714285714285715,
     'English Pale Ale': 3.483761329305137,
     'English Pale Mild Ale': 3.5952380952380953,
     'English Porter': 3.70708446866485,
     'English Stout': 3.5992647058823533,
     'English Strong Ale': 3.7560975609756095,
     'Euro Dark Lager': 3.704861111111111,
     'Euro Pale Lager': 2.9629101283880193,
     'Euro Strong Lager': 2.848024316109423,
     'Extra Special / Strong Bitter (ESB)': 3.685157421289355,
     'Flanders Oud Bruin': 3.923076923076923,
     'Flanders Red Ale': 3.25,
     'Foreign / Export Stout': 3.2545454545454544,
     'Fruit / Vegetable Beer': 3.607749077490775,
     'German Pilsener': 3.667235494880546,
     'Hefeweizen': 3.635113268608415,
     'Herbed / Spiced Beer': 3.4452054794520546,
     'Irish Dry Stout': 3.623762376237624,
     'Irish Red Ale': 2.9819277108433737,
     'Keller Bier / Zwickel Bier': 3.869565217391305,
     'Kristalweizen': 2.7857142857142856,
     'K\xc3\xb6lsch': 3.6968085106382977,
     'Lambic - Fruit': 3.75,
     'Lambic - Unblended': 3.3,
     'Light Lager': 2.39662027833002,
     'Low Alcohol Beer': 2.7142857142857144,
     'Maibock / Helles Bock': 3.7466666666666666,
     'Milk / Sweet Stout': 3.782608695652174,
     'Munich Dunkel Lager': 3.780141843971631,
     'Munich Helles Lager': 3.959230769230769,
     'M\xc3\xa4rzen / Oktoberfest': 3.5933572710951527,
     'Oatmeal Stout': 3.7745098039215685,
     'Old Ale': 4.096007604562735,
     'Pumpkin Ale': 3.7875,
     'Quadrupel (Quad)': 3.596638655462185,
     'Rauchbier': 4.067853457172355,
     'Russian Imperial Stout': 4.300371057513916,
     'Rye Beer': 4.213570634037825,
     'Saison / Farmhouse Ale': 3.702127659574468,
     'Schwarzbier': 3.6226415094339623,
     'Scotch Ale / Wee Heavy': 4.08339337175794,
     'Scottish Ale': 3.7628205128205128,
     'Scottish Gruit / Ancient Herbed Ale': 3.9076923076923076,
     'Smoked Beer': 3.19672131147541,
     'Tripel': 3.7840466926070038,
     'Vienna Lager': 3.5303030303030303,
     'Weizenbock': 3.3846153846153846,
     'Wheatwine': 4.186813186813188,
     'Winter Warmer': 3.6216216216216224,
     'Witbier': 3.5277777777777777}



### Task 2



```python
from scipy import optimize

def feature(d):
    feat = [1]
    if d['beer/style'] == 'American IPA':
        feat.append(1)
    else:
        feat.append(0)
    return feat

def f(theta, X, y, lam):
    theta = np.matrix(theta).T
    X = np.matrix(X)
    y = np.matrix(y).T

    diff = X*theta - y 
    diffSq = diff.T * diff
    diffSqReg = diffSq / len(X) + lam*(theta.T*theta)
    res = diffSqReg.flatten().tolist()
    
    return res

def fprime(theta, X, y, lam):
    theta = np.matrix(theta).T
    X = np.matrix(X)
    y = np.matrix(y).T
    diff = X*theta - y
    res = 2*X.T*diff / len(X) + 2*lam*theta
    res = np.array(res.flatten().tolist()[0])
    
    return res

X = [feature(d) for d in data]
y = [d['review/taste'] for d in data]
theta = [0,0]

res = optimize.fmin_l_bfgs_b(f, [0,0], fprime, args = (X, y, 0.1))
print "\nTheta:"
print res[0]

```

    
    Theta:
    [ 3.55048115  0.20326687]


$\theta_0 = 3.55048115, \theta_1 = 0.20326687$

$\theta_0$ represents the average for a beer that is not an American IPA

$\theta_1$ represents the extra rating a American IPA usually has compared to the average rating.

### Task 3


```python
def square_error(theta,X,y):
    theta = np.matrix(theta).T
    X = np.matrix(X)
    y = np.matrix(y).T
    diff = X*theta - y 
    diffSq = diff.T * diff
    return diffSq / len(X)

n = len(data) // 2
training_data = data[0:n]
testing_data = data[n+1::]
X_train = [feature(d) for d in training_data]
y_train = [d['review/taste'] for d in training_data]
X_test = [feature(d) for d in testing_data]
y_test = [d['review/taste'] for d in testing_data]

theta = [0, 0]
# Train on training data
theta = optimize.fmin_l_bfgs_b(f, theta, fprime, args = (X_train, y_train, 0.1))[0]
print"Theta"
print theta

print "\nMSE Training data"
print square_error(theta, X_train, y_train)

print "\nMSE Testing data"
print square_error(theta, X_test, y_test)



```

    Theta
    [ 3.53845571  0.19558705]
    
    MSE Training data
    [[ 0.68485066]]
    
    MSE Testing data
    [[ 0.61343705]]


### Task 4 


```python
styles = sorted(list(set([d['beer/style'] for d in data])))
n_styles = len(styles)
theta = [0 for x in range(n_styles+1)]
def feature_all(d, styles):
    feat = [1]
    for style in styles:
        if d['beer/style'] == style:
            feat.append(1)
        else:
            feat.append(0)
    return feat

X_train = [feature_all(d, styles) for d in training_data]
X_test = [feature_all(d,styles) for d in testing_data]
theta = optimize.fmin_l_bfgs_b(f, theta, fprime, args=(X_train, y_train, 0.1))[0]
print "Theta values"
print theta

print "\nMSE Error Training"
print square_error(theta, X_train, y_train)

print "\nMSE Error Testing"
print square_error(theta, X_test, y_test)
```

    Theta values
    [  3.31580466e+00   4.97760189e-03  -2.38587709e-02   2.51249992e-02
       1.25828376e-03   1.65675179e-01   1.25291841e-02  -1.22472466e-02
       4.49940980e-02   2.01970581e-03   2.63215217e-01   2.73431582e-04
       6.66743316e-01   2.98791917e-01  -2.84908064e-03   9.72345014e-02
      -5.55151759e-03   2.77331809e-03   1.43596748e-01   8.41384325e-02
       5.49401193e-03   3.29222951e-02   7.87718643e-03   1.32919854e-03
       2.77303108e-02   2.29748354e-02   2.13898264e-02   1.50528455e-01
       9.33064808e-04   1.71019038e-03   2.91614437e-02  -1.35025182e-02
       3.98461601e-03   1.31160036e-03   2.52554710e-03  -4.70184940e-03
      -1.18608163e-04  -5.39852574e-03  -2.35275267e-03   1.55392505e-02
       2.39893949e-03   1.38454049e-03   3.66999290e-02   1.64069539e-02
       1.91013582e-02   1.98219062e-03   9.90407293e-03  -1.29335465e-03
       1.85569918e-03   4.44073165e-02   7.88773670e-03   2.38140252e-02
       8.25781245e-03  -6.41887340e-02  -4.16106207e-02   2.75856678e-02
       3.14014931e-03   7.36655932e-05  -5.24152758e-05   1.07786623e-01
      -3.19748633e-03   3.07388227e-02  -1.27491876e-03   8.54833781e-03
      -1.07544302e-02   4.29354765e-03  -9.54505232e-04   1.24320637e-02
       1.03922841e-03  -6.22366247e-05  -1.51885536e-01  -1.82623197e-03
       6.60868349e-04   4.38876108e-03  -1.71034243e-03   1.68285791e-04
       2.40130185e-03   9.04112040e-03   9.21550842e-03   6.76837822e-02
       8.19645821e-04   1.43910861e-02   5.10752551e-01   3.03197338e-02
       1.74014707e-02   4.27008369e-03   4.02304026e-01   9.66867193e-03
       1.37409681e-02  -3.90672837e-03   2.43557485e-02   2.50773448e-03
       3.56584303e-04   9.86107064e-04   2.38677494e-02   1.13048344e-02]
    
    MSE Error Training
    [[ 0.54984543]]
    
    MSE Error Testing
    [[ 0.65791399]]


## Classification

### Task 5


```python
from sklearn import svm

def SVM_execute(X, y, C=1000):
    n = len(X)
    X_train = X[0:n//2]
    X_test = X[n//2::]
    y_train = y[0:n//2]
    y_test = y[n//2::]
    
    clf = svm.SVC(C=C)
    clf.fit(X_train, y_train)
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)

    percentage_train = sum([ y_train[i] == train_predictions[i] for i in range(n//2)]) / float(n//2)
    percentage_test =  sum([ y_test[i] == test_predictions[i] for i in range(n//2)]) / float(n//2)
    print "\nAccuracy training"
    print percentage_train
    print "\nAccuray Test-set"
    print percentage_test

X = [[d['beer/ABV'], d['review/taste']] for d in data]
y = [d['beer/style'] == 'American IPA' for d in data]
n = len(X)

SVM_execute(X, y)
```

    
    Accuracy training
    0.9226
    
    Accuray Test-set
    0.85632


### Task 6


```python
def feature_extractor(d):
    return [
        d['beer/ABV'],
        'IPA' in d['review/text']
        
    ]

X = [feature_extractor(d) for d in data]
y = [d['beer/style'] == 'American IPA' for d in data]

SVM_execute(X, y)
```

    
    Accuracy training
    0.94412
    
    Accuray Test-set
    0.95536


Using a feature vector cointaining the ABV and wether or not the review text contains 'IPA' in the review text gives the result


Accuracy training:
0.94412

Accuray Test-set:
0.95536
### Task 7


The regularization constant penalizes the complexity model. With a high regularization constant penalizes high complexity models, forcing the SVM to choose a simpler model.



```python
X = [feature_extractor(d) for d in data]
y = [d['beer/style'] == 'American IPA' for d in data]

print "Regularization constant = 0.1"
SVM_execute(X,y,0.1)

print "Regularization constant = 10"
SVM_execute(X,y,10)

print "Regularization constant = 1000"
SVM_execute(X,y,1000)

print "Regularization constant = 10000"
SVM_execute(X,y,10000)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-2f094098c7ee> in <module>()
    ----> 1 X = [feature_extractor(d) for d in data]
          2 y = [d['beer/style'] == 'American IPA' for d in data]
          3 
          4 print "Regularization constant = 0.1"
          5 SVM_execute(X,y,0.1)


    NameError: name 'data' is not defined


### Task 8


The function $f$ can be written as

$\sum_{i=1}^{n} -log(1 + e^{-x_i\theta}) - X \theta - \lambda||\theta||_2^2$

Where the second term is only considered if $y[i] = False$

The final log-likelihood is $14627.2048715$

The accuracy is $0.9218$



```python
from math import exp
from math import log
import random
import scipy
import numpy as np

def inner(x,y):
  return sum([x[i]*y[i] for i in range(len(x))])

def sigmoid(x):
  return 1.0 / (1 + exp(-x))

# NEGATIVE Log-likelihood
def f(theta, X, y, lam):
  loglikelihood = 0
  for i in range(len(X)):
    logit = inner(X[i], theta)
    loglikelihood -= log(1 + exp(-logit))
    if not y[i]:
          loglikelihood -= logit
  for k in range(len(theta)):
    loglikelihood -= lam * theta[k]*theta[k]
  return -loglikelihood

# NEGATIVE Derivative of log-likelihood
def fprime(theta, X, y, lam):
  dl = [0.0]*len(theta)
  for i in range(len(X)):
    logit = - inner(X[i], theta)
    res = np.inner(X[i], exp(logit)) / ( 1 + exp(logit))
    if not y[i]:
        res -= X[i]
    dl += res
  for i in range(len(theta)):
    dl[i] -= 2*lam*theta[i]
  # Negate the return value since we're doing gradient *ascent*
  return np.array([-x for x in dl])


X = [[d['beer/ABV'], d['review/taste']] for d in data]
y = [d['beer/style'] == 'American IPA' for d in data]

X_train = X[:len(X)/2]
X_test = X[len(X)/2:]
y_train = y[:len(X)/2]
y_test = y[len(X)/2:]

# Use a library function to run gradient descent 
theta,l,info = scipy.optimize.fmin_l_bfgs_b(f, [0]*len(X[0]), fprime, args = (X_train, y_train, 1.0))
print "Final log likelihood =", f(theta,X,y,1.0)

predictions = [sigmoid(inner(X_test[i], theta)) > 0.5 for i in range(len(X_test))]

result = [ predictions[i] == y_test[i] for i in range(len(X_test))]
acc = sum(result) / float(len(result))
print "Accuracy = ", acc
```

    Final log likelihood = 14627.2048715
    Accuracy =  0.9218

