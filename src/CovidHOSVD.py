#این فایل برای اجرای الگوریتم hosvd می باشد

# https://github.com/whistlebee/pytorch-hosvd


import os
import numpy as np
import pandas as pd
import torch
import tensorly as tl
from numpy.core.fromnumeric import shape
from typing import Tuple, List
from tensorly.random import random_tucker


tl.set_backend('pytorch')

#این متد برای درست کردن ماتریس 3 بعدی استفاده میشود و داده ها در ماتریس قرار می دهد
def FixData(tensor, items, users, df,
columns, itemColumnName, userColumnName, 
columnsForCalssification, UserIndexColumnIndex):
    matrixForCalssification = []
    print("loop:")    
    index_item = 0
    for item in items:
        index_user = 0        
        for user in users:
            print("index_item=" + str(index_item) + ", index_user=" + str(index_user), end="\r")

            query = "`"+ itemColumnName + "` == " + str(item) + " and `" + userColumnName + "` == " + str(user)
            row = df.query(query)

            user_rows = df.query(query)[columnsForCalssification].to_numpy()
            for user_row in user_rows:
                user_row[UserIndexColumnIndex] = index_user
            matrixForCalssification.extend(user_rows)
            
            #print(shape(matrixForCalssification))
            #print(len(matrixForCalssification))            

            index_column = 0
            for column in columns:
                tensor[index_item, index_user, index_column] = -1
                if row.empty == False:
                    tensor[index_item, index_user, index_column] = row[column].values[0]
                index_column = index_column + 1                                        
            index_user = index_user + 1
        index_item = index_item + 1
    print("end loop")

    # print(tensor)
    # print(shape(tensor))
    # print(shape(tensor[0]))
    # print(shape(tensor[0][0]))
    return tensor, matrixForCalssification

#این متد الگوریتم svd را اجرا میکند
def truncated_svd(
    A: torch.Tensor,
    k: int,
    n_iter: int = 2,
    n_oversamples: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the Truncated SVD.
    Based on fbpca's version.
    Parameters
    ----------
    A : (M, N) torch.Tensor
    k : int
    n_iter : int
    n_oversamples : int
    Returns
    -------
    u : (M, k) torch.Tensor
    s : (k,) torch.Tensor
    vt : (k, N) torch.Tensor
    """
    m, n = A.shape
    Q = torch.randn(n, k + n_oversamples)
    Q = A @ Q

    Q, _ = torch.linalg.qr(Q)

    # Power iterations
    for _ in range(n_iter):
        Q = (Q.t() @ A).t()
        Q, _ = torch.linalg.qr(Q)
        Q = A @ Q
        Q, _ = torch.linalg.qr(Q)

    QA = Q.t() @ A
    # Transpose QA to make it tall-skinny as MAGMA has optimisations for this
    # (USVt)t = VStUt
    Va, s, R = torch.linalg.svd(QA.t(), full_matrices=False)
    U = Q @ R.t()

    return U[:, :k], s[:k], Va.t()[:k, :]

#این متد الگوریتم hosvd اجرا میکند
def sthosvd( tensor: torch.Tensor, core_size: List[int] ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Seqeuntially Truncated Higher Order SVD.
    Parameters
    ----------
    tensor : torch.Tensor,
        arbitrarily dimensional tensor
    core_size : list of int
    Returns
    -------
    torch.Tensor
        core tensor
    List[torch.Tensor]
        list of singular vectors
    List[torch.Tensor]
        list of singular vectors
    """
    intermediate = tensor
    singular_vectors, singular_values = [], []
    for mode in range(len(tensor.shape)):
        to_unfold = intermediate
        svec, sval, _ = truncated_svd(
            tl.unfold(to_unfold, mode), core_size[mode])
        intermediate = tl.tenalg.mode_dot(intermediate, svec.t(), mode)
        singular_vectors.append(svec)
        singular_values.append(sval)
    return intermediate, singular_vectors, singular_values

#این متد یک لیست را یکتا میکند
def ListHelperUnique(List):
    unique_list = []
    for x in List:
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

#pid	Overall	reliability	quality	related	useful	user_index


#در اینجا ابتدا دیتا ها را از فایل های csv استخراج میکنیم
df = pd.read_csv("{0}\data\{1}".format(os.getcwd(), "ansari.covid.csv"))

allColumns = ["pid", "user_index", "reliability", "quality", "related", "useful"]

columnsCriterion = ["reliability", "quality", "related", "useful"]

columnsForCalssification = ["pid", "user_index",
    "reliability", "quality", "related", "useful",     
    "Overall"]

UserIndexColumnIndex = 1

overallScoreColumnName = "Overall"

items = ListHelperUnique(df["pid"].to_numpy())
users = ListHelperUnique(df["user_index"].to_numpy())

print("rows : ", shape(df[allColumns].to_numpy()))
print("users : ", shape(users))
print("items : ", shape(items))



tensorSize = (len(items), len(users), len(columnsCriterion))
tensor = torch.empty(tensorSize)
print("tensor : ", shape(tensor))


#ساخت ماتریس 3 بعدی
tensor, matrixForCalssification = FixData(tensor, items, users, df,
    columnsCriterion, allColumns[0], allColumns[1], 
    columnsForCalssification, UserIndexColumnIndex)

#tensor = random_tucker(tensorSize, rank=tensorSize, full=True)

#اجرا الگوریتم hosvd
intermediate, singular_vectors, singular_values = sthosvd(tensor, tensorSize)

print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("intermediate:")
print(shape(intermediate))
print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("singular_vectors:")
print(len(singular_vectors))
print(shape(singular_vectors[0]))
print(shape(singular_vectors[1]))
print(shape(singular_vectors[1][0]))
print(shape(singular_vectors[2]))
print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("singular_values:")
print(len(singular_values))
print(shape(singular_values[0]))
print(shape(singular_values[1]))
print(shape(singular_values[2]))
print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")
print("XXXXXXXXXXXXXXXXXXXXXXXXXXX")

#ذخیره کردن نتیجه الگوریتم hosvd که در فایل بعدی مورد استفاده 
path = "{0}\data\singular_vectors_covid_reviews.txt".format(os.getcwd())
np.savetxt(path, singular_vectors[1])
singular_vectors_hotel_reviews = np.loadtxt(path)

path = "{0}\data\matrixForCalssification_covid.txt".format(os.getcwd())
np.savetxt(path, matrixForCalssification)
matrixForCalssification = np.loadtxt(path)

print(len(singular_vectors_hotel_reviews))
print(shape(singular_vectors_hotel_reviews))

print(len(matrixForCalssification))
print(shape(matrixForCalssification))

print("======================HOSVD IsReady=================================")
