# CF-LSH-HE
*Alpha Version*

[中文](README_zh-CN.md)

## Overview
This is an example of a combination of **Collaborative Filtering**(CF), **Local Sensitive Hashing**(LSH), and **Homomorphic Encryption**(HE) algorithms. The project focuses on how to combine real-time recommendations with multiple data in a relatively secure environment.

The core idea of the implementation lies in the fact that, based on the traditional collaborative filtering algorithm, the index is generated offline through local sensitive hashing to reduce the amount of calculation involved in the recommendation when the recommended target arrives, enhances real-time performance, and makes use of Paillier's properties of homomorphic encryption to make When the data of all parties is **DIRECTLY** exposed, multiple parties are allowed to participate in the recommended calculation to improve the accuracy of the recommendation results.

### Collaborative Filtering
The collaborative filtering algorithm uses the user's historical behavior data to find similar individuals, and predicts items that the user does not generate data through similarity, rating, and other information to obtain recommended results.

### Local Sensitive Hashing
A localized sensitive hashing algorithm is a technology that can map the objects that **may be** approximated in a high dimensional space to the same hash bucket **with a maximum probability** after dimensionality reduction. In the recommendation algorithm, this technique helps us to reduce the number of users or individuals involved in the calculation without having to traverse the entire data set.

### Homomorphic Encryption
The homomorphic encryption algorithm is aimed at a cryptographic algorithm that satisfies a certain characteristic, so that we can directly calculate the ciphertext, and the value obtained after the calculation result is decrypted is the same as the result of directly calculating the plaintext. With homomorphic encryption, we are expected to participate in computing without touching the original plaintext data. The Paillier algorithm used in this project has additive homomorphism properties. I mentioned "direct" in the foregoing because I did not delve deeper into the possibility that the malicious participant could try to reverse the original data by using a large amount of counterfeit data to continuously probe the calculation results. Therefore, the possibility of this situation cannot be ruled out. . If you understand this, welcome to contact me and discuss.
