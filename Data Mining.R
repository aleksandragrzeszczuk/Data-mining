# Wczytujemy biblioteki
library(tidyverse)
library(DataExplorer)
library(xtable)
library(dlookr)
library(MASS)
library(forcats)
library(summarytools)
library(ggplot2)
library(egg)
library(pROC)
library(GGally)
library(caret)
library(klaR)
library(class)
library(prediction)
library(neuralnet)

# Ustalamy ścieżkę do pliku
# getwd()
setwd("C:/Users/olagr/Desktop/DataMining")

# Wczytujemy dane
german.data <- read.csv("german.data.csv", sep = "")
dane <- german.data
# view(dane)

# german.data.numeric <- read.csv("german.data-numeric.csv", sep = "")

########################################
########################################
# EKSPLORACYJNA ANALIZA DANYCH
########################################
########################################

# Zmienamy nazwy kolumn
colnames(dane) <- c("Status of existing checking account","Duration", "Credit history", "Purpose", "Amount", "Savings account and bonds", "Present employment since", "Installment rate in percentage of disposable income", "Personal status and sex", "Other debtors or guarantors", "Present residence since", "Property", "Age", "Other installment plans", "Housing", "Number of existing credits at this bank", "Job", "Number of people being liable to provide maintenance for", "Telephone", "Foreign worker", "Response")
colnames(dane) <- c("chk_acct","duration","credit_his","purpose","amount","saving_acct","present_emp","installment_rate","sex","other_debtor","present_resid","property","age","other_install","housing","n_credits","job","n_people","telephone","foreign","response")

view(dane)

# Sprawdzamy nasze dane
nrow(german.data)
ncol(german.data)
str(german.data)
glimpse(german.data)

# Podsumowanie danych
xtable(summary(german.data))

# Sprawdzamy podsumowanie zmiennych
metadata <- t(introduce(german.data))
colnames(metadata)<-"Values"
xtable(metadata)

plot_intro(german.data)

# Statystyki opiowe dla zmiennych ilościowych
opis <- describe(dane)
xtable(opis)

var(dane$duration)
var(dane$amount)
var(dane$installment_rate)
var(dane$present_resid)
var(dane$age)
var(dane$n_credits)
var(dane$n_people)
var(dane$response)

# Analiza korelacji
cor <- correlate(dane)
xtable(cor)

# Wykresy pudełkowe wszystkich zmiennych ilościowych
duration <- boxplot(dane$duration, main = "duration", col = "#FFDF00")
amount <- boxplot(dane$amount, main = "amount", col = "#FF9933")
installment <- boxplot(dane$installment_rate, main = "installment rate", col = "#FF0800")
present <- boxplot(dane$present_resid, main = "present resid", col = "#BD33A4")
age <- boxplot(dane$age, main = "age", col = "#702963")
credits <- boxplot(dane$n_credits, main = "n credits", col = "#333399")
people <- boxplot(dane$n_people, main = "n people", col = "#A2ADD0")
response <- boxplot(dane$response, main = "response", col = "#6699CC")

# Histogramy zmiennych ilościowych
plot_histogram(dane, ncol = 3L)

# Gęstości empiryczne zmiennych ilościowych
plot_density(dane, ncol = 3L)

# Wykresy kwantylowe zmiennych ilościowych 
plot_qq(dane, ncol = 4L)

# Wykresy słupkowe zmiennych jakościowych
plot_bar(dane)

#
plot_scatterplot(dane, by = "dane$response")

# Stosunek dobych i złych klientów do danych kredytowych
X <- pur <- dane %>% group_by(dane$credit_his) %>% count(dane$response)
xtable(X)

# Stosunek dobrych i złych klientów do płci i statustu społecznego
ggplot(dane, aes(sex, ..count..)) + 
  geom_bar(aes(fill = as.factor(response)), position = "dodge") +
  scale_fill_discrete(labels=c('Good', 'Bad'))

# Stosunek dobrych i złych klientów saving_acct
ggplot(dane, aes(saving_acct, ..count..)) + 
  geom_bar(aes(fill = as.factor(response)), position = "dodge") +
  scale_fill_discrete(labels=c('Good', 'Bad'))

# Stosunek dobrych i złych klientów chk_acct
ggplot(dane, aes(chk_acct, ..count..)) + 
  geom_bar(aes(fill = as.factor(response)), position = "dodge") +
  scale_fill_discrete(labels=c('Good', 'Bad'))

################################################################################
################################################################################
# KLASYFIKACJI WRAZ Z OCENĄ DOKŁADNOŚCI
################################################################################
################################################################################

# Wczytujemy dane
german.data <- read.csv("german.data.csv", sep = "")

colnames(german.data) <- c("chk_acct","duration","credit_his","purpose","amount","saving_acct","present_emp","installment_rate","sex","other_debtor","present_resid","property","age","other_install","housing","n_credits","job","n_people","telephone","foreign","response")

# Funkcja ifelse sprawdza, czy wartość w kolumnie response jest równa 2. Jeśli tak to zastępuje ją wartością 0 
# W przeciwnym razie wartością 1
# Czyli teraz mamy, że dobry klient jest oznaczony 1, zaś zły klient - wartością 0 
german.data$response <- ifelse(german.data$response == 2, 0, 1)

# view(german.data)

set.seed(255707)

a <- german.data[,-8]
b <- a[,-10]
c <- b[,-14]
german.data <- c[,-15]

# Wybieramy losowo 2/3 wszystkich obserwacji do zbioru uczącego, pozostałe do testowefi 
n <- dim(german.data)[1]
prop <- 2/3

learning.indx <- sample(1:n,prop*n)
learning.set <- german.data[learning.indx,]
# view(learning.set)
test.set <- german.data[-learning.indx,]

# Sprawdzamy, czy dobrze rozdzieliśmy dane
# summary(learning.set)
# summary(test.set)

# Wybieramy kilka modeli, różnymi metodami, które będziemy analizować

# Wybór zmiennych w opraciu o model regresji logistycznej 
logistic <- glm(response~., data = learning.set)
# Backward AIC
logistic_AIC <- step(logistic)
summary(logistic_AIC)
# Model: response ~ chk_acct + duration + credit_his + purpose + saving_acct + present_emp + other_debtor + age + other_install + housing + foreign
# Backward BIC
logistic_BIC <- step(logistic, k=log(nrow(learning.set))) 
summary(logistic_BIC)
# Model: response ~ chk_acct + duration + credit_his

# Wybór zmiennych w oparciu o bibliotekę klaR i metodę krokową
lda.forward.selection <- stepclass(response~., data=learning.set, method="lda", direction="forward", improvement=0.01)
# Model: response ~ duration
qda.forward.selection <- stepclass(response~., data=learning.set, method="qda", direction="forward", improvement=0.01)
# Model: response ~ duration + amount

# Czyli ostatecznie mamy 5 modeli 
# Model M1 -> response~.
# Model M2 -> response ~ chk_acct + duration + credit_his + purpose + saving_acct + present_emp + other_debtor + age + other_install + housing + foreign
# Model M3 -> response ~ chk_acct + duration + credit_his
# Model M4 -> response ~ duration
# Model M5 -> response ~ duration + amount

# Metoda LDA - LINIOWA ANALIZA DYSKRYMINACYJNA

# Konstrukcja reguły klasyfikacyjnej dla wszystkich zmiennych 
dane.lda <- lda(response~., data = german.data, subset = learning.indx)

# Konstrukcja reguły klasyfikacyjnej dla wybranych zmiennych 
dane.lda.choosen1 <- lda(response ~ chk_acct + duration + credit_his + purpose + saving_acct + present_emp + other_debtor + age + other_install + housing + foreign, data = german.data, subset = learning.indx)
dane.lda.choosen2 <- lda(response ~ chk_acct + duration + credit_his, data = german.data, subset = learning.indx)
dane.lda.choosen3 <- lda(response ~ duration, data = german.data, subset = learning.indx)
dane.lda.choosen4 <- lda(response ~ duration + amount, data = german.data, subset = learning.indx)

# Prognozy dla zbioru testowego 
prognozy.lda <-  predict(dane.lda, test.set)
prognozy.lda.choosen1 <-  predict(dane.lda.choosen1, test.set)
prognozy.lda.choosen2 <-  predict(dane.lda.choosen2, test.set)
prognozy.lda.choosen3 <-  predict(dane.lda.choosen3, test.set)
prognozy.lda.choosen4 <-  predict(dane.lda.choosen4, test.set)

# Prawdopodobieństwo a posteriori
prognozy.lda.prob <- prognozy.lda$posterior
prognozy.lda.choosen1.prob <- prognozy.lda.choosen1$posterior
prognozy.lda.choosen2.prob <- prognozy.lda.choosen2$posterior
prognozy.lda.choosen3.prob <- prognozy.lda.choosen3$posterior
prognozy.lda.choosen4.prob <- prognozy.lda.choosen4$posterior

# Prognozowane etykietki 
etykietki.lda <- prognozy.lda$class
etykietki.lda.choosen1 <- prognozy.lda.choosen1$class
etykietki.lda.choosen2 <- prognozy.lda.choosen2$class
etykietki.lda.choosen3 <- prognozy.lda.choosen3$class
etykietki.lda.choosen4 <- prognozy.lda.choosen4$class

# Ocena dokładności klasyfikacji
rzeczywiste <- german.data$response[-learning.indx]
xtable(conf.mat.lda <- table(etykietki.lda, rzeczywiste))
xtable(conf.mat.lda.choosen1 <- table(etykietki.lda.choosen1, rzeczywiste))
xtable(conf.mat.lda.choosen2 <- table(etykietki.lda.choosen2, rzeczywiste))
xtable(conf.mat.lda.choosen3 <- table(etykietki.lda.choosen3, rzeczywiste))
xtable(conf.mat.lda.choosen4 <- table(etykietki.lda.choosen4, rzeczywiste))

# Błąd klasyfikacji na zbiorze testowym
n.test <- dim(test.set)[1]
(blad.lda <- (n.test-sum(diag(conf.mat.lda))) /n.test)
(blad.lda.choosen1 <- (n.test-sum(diag(conf.mat.lda.choosen1))) /n.test)
(blad.lda.choosen2 <- (n.test-sum(diag(conf.mat.lda.choosen2))) /n.test)
(blad.lda.choosen3 <- (n.test-sum(diag(conf.mat.lda.choosen3))) /n.test)
(blad.lda.choosen4 <- (n.test-sum(diag(conf.mat.lda.choosen4))) /n.test)

# Krzywe ROC 
# Model_M1
prob_M1 <- predict(dane.lda, newdata = test.set)$posterior[, 2]
roc_M1 <- roc(test.set$response, prob_M1)
auc(roc_M1)

# Model_M2
prob_M2 <- predict(dane.lda.choosen1, newdata = test.set)$posterior[, 2]
roc_M2 <- roc(test.set$response, prob_M2)
auc(roc_M2)

# Model_M3
prob_M3 <- predict(dane.lda.choosen2, newdata = test.set)$posterior[, 2]
roc_M3 <- roc(test.set$response, prob_M3)
auc(roc_M3)

# Model_M4
prob_M4 <- predict(dane.lda.choosen3, newdata = test.set)$posterior[, 2]
roc_M4 <- roc(test.set$response, prob_M4)
auc(roc_M4)

# Model_M5
prob_M5 <- predict(dane.lda.choosen4, newdata = test.set)$posterior[, 2]
roc_M5 <- roc(test.set$response, prob_M5)
auc(roc_M5)

par(mfrow = c(1, 3))
plot(roc_M1, col = "#9F00FF", main = "Krzywa ROC M1")
plot(roc_M2, col = "#D3003F", main = "Krzywa ROC M2")
plot(roc_M3, col = "#0000CD", main = "Krzywa ROC M3")

par(mfrow = c(1, 2))
plot(roc_M4, col = "#00416A", main = "Krzywa ROC M4")
plot(roc_M5, col = "#614051", main = "Krzywa ROC M5")

par(mfrow = c(1, 1))

library(caret)

# Tworzenie predykcji dla każdego modelu
Model_M1 <- glm(response~., data = learning.set)
Model_M2 <- step(Model_M1) 
Model_M3 <- step(Model_M1, k=log(nrow(learning.set))) 
Model_M4 <- glm(response ~ duration, data = learning.set)
Model_M5 <- glm(response ~ duration + amount, data = learning.set)

predicted_M1 <- predict(Model_M1, newdata = test.set, type = "response")
predicted_M2 <- predict(Model_M2, newdata = test.set, type = "response")
predicted_M3 <- predict(Model_M3, newdata = test.set, type = "response")
predicted_M4 <- predict(Model_M4, newdata = test.set, type = "response")
predicted_M5 <- predict(Model_M5, newdata = test.set, type = "response")

# Tworzenie macierzy pomyłek 
confusion_M1 <- confusionMatrix(as.factor(predicted_M1 > 0.5), as.factor(test.set$response > 0.5))
confusion_M2 <- confusionMatrix(as.factor(predicted_M2 > 0.5), as.factor(test.set$response > 0.5))
confusion_M3 <- confusionMatrix(as.factor(predicted_M3 > 0.5), as.factor(test.set$response > 0.5))
confusion_M4 <- confusionMatrix(as.factor(predicted_M4 > 0.5), as.factor(test.set$response > 0.5))
confusion_M5 <- confusionMatrix(as.factor(predicted_M5 > 0.5), as.factor(test.set$response > 0.5))

# Mteoda QDA - KWADRATOWA ANALIZA DYSKRYMINACTJNA

# Konstrukcja reguły klasyfikacyjnej dla wszystkich zmiennych 
dane.qda <- qda(response~., data = german.data, subset = learning.indx)

# Konstrukcja reguły klasyfikacyjnej dla wybranych zmiennych 
dane.qda.choosen1 <- qda(response ~ chk_acct + duration + credit_his + purpose + saving_acct + present_emp + other_debtor + age + other_install + housing + foreign, data = german.data, subset = learning.indx)
dane.qda.choosen2 <- qda(response ~ chk_acct + duration + credit_his, data = german.data, subset = learning.indx)
dane.qda.choosen3 <- qda(response ~ duration, data = german.data, subset = learning.indx)
dane.qda.choosen4 <- qda(response ~ duration + amount, data = german.data, subset = learning.indx)

# Prognozy dla zbioru testowego 
prognozy.qda <-  predict(dane.qda, test.set)
prognozy.qda.choosen1 <-  predict(dane.qda.choosen1, test.set)
prognozy.qda.choosen2 <-  predict(dane.qda.choosen2, test.set)
prognozy.qda.choosen3 <-  predict(dane.qda.choosen3, test.set)
prognozy.qda.choosen4 <-  predict(dane.qda.choosen4, test.set)

# Prawdopodobieństwo a posteriori
prognozy.qda.prob <- prognozy.qda$posterior
prognozy.qda.choosen1.prob <- prognozy.qda.choosen1$posterior
prognozy.qda.choosen2.prob <- prognozy.qda.choosen2$posterior
prognozy.qda.choosen3.prob <- prognozy.qda.choosen3$posterior
prognozy.qda.choosen4.prob <- prognozy.qda.choosen4$posterior

# Prognozowane etykietki 
etykietki.qda <- prognozy.qda$class
etykietki.qda.choosen1 <- prognozy.qda.choosen1$class
etykietki.qda.choosen2 <- prognozy.qda.choosen2$class
etykietki.qda.choosen3 <- prognozy.qda.choosen3$class
etykietki.qda.choosen4 <- prognozy.qda.choosen4$class

# Ocena dokładności klasyfikacji 
(conf.mat.qda <- table(etykietki.qda, rzeczywiste))
(conf.mat.qda.choosen1 <- table(etykietki.qda.choosen1, rzeczywiste))
(conf.mat.qda.choosen2 <- table(etykietki.qda.choosen2, rzeczywiste))
(conf.mat.qda.choosen3 <- table(etykietki.qda.choosen3, rzeczywiste))
(conf.mat.qda.choosen4 <- table(etykietki.qda.choosen4, rzeczywiste))

# Błąd klasyfikacji na zbiorze testowym
(blad.qda <- (n.test-sum(diag(conf.mat.qda))) /n.test)
(blad.qda.choosen1 <- (n.test-sum(diag(conf.mat.qda.choosen1))) /n.test)
(blad.qda.choosen2 <- (n.test-sum(diag(conf.mat.qda.choosen2))) /n.test)
(blad.qda.choosen3 <- (n.test-sum(diag(conf.mat.qda.choosen3))) /n.test)
(blad.qda.choosen4 <- (n.test-sum(diag(conf.mat.qda.choosen4))) /n.test)

# Krzywe ROC 
# Model_M1
prob_M1_qda <- predict(dane.qda, newdata = test.set)$posterior[, 2]
roc_M1 <- roc(test.set$response, prob_M1)
auc(roc_M1)

# Model_M2
prob_M2 <- predict(dane.qda.choosen1, newdata = test.set)$posterior[, 2]
roc_M2 <- roc(test.set$response, prob_M2)
auc(roc_M2)

# Model_M3
prob_M3 <- predict(dane.qda.choosen2, newdata = test.set)$posterior[, 2]
roc_M3 <- roc(test.set$response, prob_M3)
auc(roc_M3)

# Model_M4
prob_M4 <- predict(dane.qda.choosen3, newdata = test.set)$posterior[, 2]
roc_M4 <- roc(test.set$response, prob_M4)
auc(roc_M4)

# Model_M5
prob_M5 <- predict(dane.qda.choosen4, newdata = test.set)$posterior[, 2]
roc_M5 <- roc(test.set$response, prob_M5)
auc(roc_M5)

par(mfrow = c(1, 3))
plot(roc_M1, col = "#9F00FF", main = "Krzywa ROC M1")
plot(roc_M2, col = "#D3003F", main = "Krzywa ROC M2")
plot(roc_M3, col = "#0000CD", main = "Krzywa ROC M3")

par(mfrow = c(1, 2))
plot(roc_M4, col = "#00416A", main = "Krzywa ROC M4")
plot(roc_M5, col = "#614051", main = "Krzywa ROC M5")

par(mfrow = c(1, 1))

# Metoda k - NN (NAJBLIŻSZYCH SĄSIADÓW)

# losowanie podzbiorów
n <- dim(german.data)[1]

view(german.data)

# losujemy obiekty do zbioru uczącego i testowego
learning.set.index <- sample(1:n,2/3*n)

# tworzymy zbiór uczący i testowy
learning.set <- german.data[learning.set.index,]
test.set     <- german.data[-learning.set.index,]

# rzeczywiste odpowiedzi
etykietki.rzecz <- test.set$response

# teraz robimy prognozę
etykietki.prog <- knn(learning.set[,-17], test.set[,-17], learning.set$response, k=5)

view(learning.set)
# tablica kontyngencji
(wynik.tablica <- table(etykietki.prog,etykietki.rzecz))

# błąd klasyfikacji
n.test <- dim(test.set)[1]
(n.test - sum(diag(wynik.tablica))) / n.test

# Ustawienie parametru k za pomocą krzyżowej walidacji
set.seed(255707)
kfold <- trainControl(method = "cv", number = 10)
knn_model <- train(response ~ ., data = learning.set, method = "knn", tuneLength = 10, trControl = kfold)
print(knn_model)

# teraz robimy prognozę
etykietki.prog <- knn(learning.set[,-17], test.set[,-17], learning.set$response, k=23)

# tablica kontyngencji
(wynik.tablica <- table(etykietki.prog,etykietki.rzecz))

# błąd klasyfikacji
n.test <- dim(test.set)[1]
(n.test - sum(diag(wynik.tablica))) / n.test

# Sieć neuronowa
credit <- read.csv("german.data.csv", sep = "")


# zmiana nazw kolumn
names(credit) <- c("checking_account", "duration", "credit_history", "purpose", "credit_amount", "savings_account",
                   "employment_duration", "installment_rate", "personal_status_sex", "other_debtors", "present_residence",
                   "property", "age", "other_installment_plans", "housing", "number_credits", "job", "people_liable",
                   "telephone", "foreign_worker", "response")

cat_cols <- c("checking_account", "credit_history", "purpose", "savings_account",
              "employment_duration", "personal_status_sex", "other_debtors",
              "property", "other_installment_plans", "housing", "job",
              "telephone", "foreign_worker")

# konwertujemy zmienne kategoryczne na numeryczne
for (col in cat_cols) {
  credit[[col]] <- as.numeric(factor(credit[[col]]))
}

# podział danych na zbiór treningowy i testowy
n <- dim(credit)[1]
prop <- 2/3

learning.indx <- sample(1:n, prop * n)
learning.set <- credit[learning.indx, ]
test.set <- credit[-learning.indx, ]

# standaryzacja danych
preproc <- preProcess(learning.set, method = c("center", "scale"), useNA = "ifany")
trainTransformed <- predict(preproc, learning.set)
testTransformed <- predict(preproc, test.set)

# trenowanie modelu sieci neuronowej
nn <- neuralnet(response ~ ., data = trainTransformed, hidden = c(5, 3), linear.output = FALSE)

# predykcja na zbiorze testowym
pred <- predict(nn, testTransformed, type = "class")

# zamiana wyników na wartości binarne
predBinary <- ifelse(pred > 0.5, 1, 0)

# unique(predBinary)
# unique(test.set$response)

# konwertowanie test.set$response na zmienne numeryczne
test.set$response <- as.numeric(test.set$response) - 1

# wyliczenie macierzy pomyłek i miar jakości modelu
confusionMatrix(table(predBinary, test.set$response), dnn = c("predicted", "actual"))

