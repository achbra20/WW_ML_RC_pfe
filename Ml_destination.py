import requests
import pandas as pd
import json
import os
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import vincenty

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import cross_val_score

import scipy.sparse as sparse
import random
import implicit

class Etl_Model_Ml_Destination:

    @staticmethod
    def web_service_response(url):
        print("log api")
        response = requests.get(url)
        data= response.json()
        return  pd.DataFrame(data)

    @staticmethod
    def separation(lists,champ,label,sp):
        print("log separetion")
        speration = []
        b = True
        for index, request in lists.iterrows():
            if request[champ] != "":
                x = request[champ].split(sp)
                for i in range(len(x)):
                    speration.append({label: x[i], "year": request["year"], "month": request["month"],
                                        "day": request["day"]})
        return pd.DataFrame(speration)

    @staticmethod
    def found_id(list_id, champ_p, champ_id, list_req, champ_req):
        lists = []
        for index, request in list_req.iterrows():
            for index, req in list_id.iterrows():
                if req[champ_p].upper() in request[champ_req].upper() and req[champ_p] != "":
                    lists.append(
                        {'label': req[champ_p], 'destination': request[champ_req],'id': req[champ_id] , 'id_par': req["parent_location"],
                         "type": req["location_type"]})
                    break
        return lists

    @staticmethod
    def writeToJSONFile(path, fileName, data):
        filePathNameWExt = path + '/' + fileName + '.json'
        if os.path.isfile(filePathNameWExt):
            os.remove(filePathNameWExt)
        with open(filePathNameWExt, 'w') as fp:
            json.dump(data, fp)

    @staticmethod
    def open_json(file,path):
        with open(path+"/"+file+'.json') as json_data:
            data_dict = json.load(json_data)
        return pd.DataFrame(data_dict)

class  Creation_Model:

    # todo count the number of reservation for destination in dd/mm/yyyy
    def ranked(self,lists):
        rank_res = []
        unique = list(lists.destination.unique())
        for i in range(0, len(unique)):
            one_dest = pd.DataFrame(lists[lists["destination"] == unique[i]])
            result = []
            for index, request in one_dest.iterrows():
                notfound = True
                for j in range(0, len(result)):
                    if result[j]["year"] == request["year"] and result[j]["month"] == request["month"] and \
                            result[j]["day"] == request["day"]:
                        result[j]["counts"] = result[j]["counts"] + 1
                        notfound = False
                        break
                if len(result) == 0 or notfound == True:
                    result.append({"destination": request["destination"], "counts": 1, "year": request["year"],
                                         "month": request["month"], "day": request["day"]})
            rank_res = rank_res + result
        rank_res = sorted(rank_res, key=lambda k: k["year"], reverse=True)
        return rank_res

    #todo found the parent id for a destination and put it in dictionary
    def found_id_parent(self,all_destination,model_destination_all):
        model_destination_all = pd.DataFrame(model_destination_all)
        uniqueid = model_destination_all.drop_duplicates(subset="destination", keep="first")
        destenation_id = Etl_Model_Ml_Destination.found_id(all_destination, "location_name", "location_id", uniqueid, "destination")
        model_destination = pd.merge(pd.DataFrame(model_destination_all), pd.DataFrame(destenation_id), on='destination', how='inner')
        final = []
        all_destination_lf = pd.DataFrame(all_destination[all_destination["ref_language"] == "1"])
        for index, request in model_destination.iterrows():
            if request["type"] == "country" or request["type"] == "ocean":
                final.append(
                    {'id': request['id'], "year": request["year"], 'month': request["month"], 'day': request["day"],
                     "counts": request["counts"]})
            else:
                one_dest = pd.DataFrame(all_destination_lf[all_destination_lf["location_id"] == request['id_par']])
                for index, req in one_dest.iterrows():
                    if req["location_type"] == "zone":
                        final.append({'id': req['parent_location'], "year": request["year"], 'month': request["month"],
                                      'day': request["day"], "counts": request["counts"]})
                    else:
                        final.append({'id': req["location_id"], "year": request["year"], 'month': request["month"],
                                      'day': request["day"], "counts": request["counts"]})
        return final

    #todo split the composed  destination in dataframe
    def separtion(self,model_destination):
        model_destination = model_destination[model_destination["month"] != "0"]
        model_destination = Etl_Model_Ml_Destination.separation(model_destination, "request_destination", "destination", "/")
        model_destination = Etl_Model_Ml_Destination.separation(model_destination, "destination", "destination", ",")
        model_destination = pd.DataFrame(model_destination)
        model_destination = model_destination[model_destination["destination"] != ""]
        return model_destination

    #todo create the dataframe for training
    def create_model(self,api_ww,api_crm,api_all_destination,root_name):

        data_ww           = Etl_Model_Ml_Destination.web_service_response(api_ww)
        data_crm          = Etl_Model_Ml_Destination.web_service_response(api_crm)
        all_destination   = Etl_Model_Ml_Destination.web_service_response(api_all_destination)

        df_destination    = data_ww.append(data_crm)

        df_destination    = self.separtion(df_destination)

        model_destination = self.ranked(df_destination)

        model_destination = self.found_id_parent(all_destination,model_destination)

        Etl_Model_Ml_Destination.writeToJSONFile(root_name, "df_destination_id", model_destination)

        print("Json created")

    #todo training data and save the model
    def trainig_model(self,root_name,name_model):
        model_destination = Etl_Model_Ml_Destination.open_json("df_destination_id",root_name)
        df_model = pd.DataFrame(model_destination)

        feature_col_names     = ['year', 'month', 'day' , 'id']
        predicted_class_names = ['counts']
        X                     = df_model[feature_col_names].values
        y                     = df_model[predicted_class_names].values

        split_test_size       = 0.30

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42)

        model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=7)
        model.fit(X_train, y_train)

        joblib.dump(model,root_name+"/"+name_model)


        scores = cross_val_score(model, X_train, y_train, cv=5)
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

        print(model.score(X_train, y_train))
        print(model.score(X_test, y_test))
        x_pred = model.predict(X_train)
        print(mean_squared_error(y_train, x_pred))
        y_pred = model.predict(X_test)
        print(mean_squared_error(y_test, y_pred))
        ####################
        predection = Predection_distination(root_name+"/"+name_model)
        predection.restart(root_name+"/"+name_model)
        #####################################
        print ("training don !")

def singleton(theClass):
    """ decorator for a class to make a singleton out of it """
    classInstances = {}
    def getInstance(*args, **kwargs):
        """ creating or just return the one and only class instance.
            The singleton depends on the parameters used in __init__ """
        key = (theClass, args, str(kwargs))
        if key not in classInstances:
            classInstances[key] = theClass(*args, **kwargs)
        return classInstances[key]
    return getInstance

@singleton
class Predection_distination():

    def __init__(self,name):
        self.model = joblib.load(name)

    def restart(self,name):
        self.model = joblib.load(name)
        print(name)

    def predict(self,year, month, day , id):
        X_new = [[year, month, day , id]]
        knr = self.model.predict(X_new)
        return knr

class Creation_Model_recommendation:

    def rank(self,lists):
        res_des = []
        unique = list(lists.id.unique())
        for i in range(0, len(unique)):
            one_dest = pd.DataFrame(lists[lists["id"] == unique[i]])
            res_id_des = []
            for index, request in one_dest.iterrows():
                notfound = True
                for j in range(0, len(res_id_des)):
                    if request["country"] == res_id_des[j]["country"]:
                        res_id_des[j]["counts"] = res_id_des[j]["counts"] + 1
                        notfound = False
                        break
                if len(res_id_des) == 0 or notfound == True:
                    res_id_des.append({"id":request["id"],"destination":request["destination"], "counts": 1, "country": request["country"]})
            res_des = res_des + res_id_des
        res_des = sorted(res_des, key=lambda k: k["counts"], reverse=True)
        return res_des

    def found_country(self,all_destination,destination_id):
        print("log final country")
        final = []
        all_destination_lf = pd.DataFrame(all_destination[all_destination["ref_language"] == "1"])
        for index, request in destination_id.iterrows():
            if request["type"].upper() == "COUNTRY" or request["type"].upper() == "OCEAN":
                one_dest = pd.DataFrame(all_destination_lf[all_destination_lf["location_id"] == request['id']])
                for index, req in one_dest.iterrows():
                    final.append({'id': request['id'],"destination":req["location_name"], 'country': request["country"]})
            elif request["type"].upper() == "ZONE" :
                one_dest = pd.DataFrame(all_destination_lf[all_destination_lf["location_id"] == request['id_par']])
                for index, req in one_dest.iterrows():
                    final.append({'id': req['location_id'],"destination":req["location_name"], 'country': request["country"]})
            elif request["type"].upper() == "CITY" :
                one_dest = pd.DataFrame(all_destination_lf[all_destination_lf["location_id"] == request['id_par']])
                for index, req in one_dest.iterrows():
                    if req["location_type"].upper() == "ZONE":
                        one_dest2 = pd.DataFrame(all_destination_lf[all_destination_lf["location_id"] == request['id_par']])
                        for index, req2 in one_dest2.iterrows():
                            final.append({'id': req2['location_id'], "destination": req2["location_name"],'country': request["country"]})


        return final

    def do_separation(self,list, champ, country, sp):
        Statistique = []
        b = True
        for index, request in list.iterrows():
            if request[champ] != "":
                x = request[champ].split(sp)
                for i in range(len(x)):
                    Statistique.append({"destination": x[i], "country": request[country]})
        return pd.DataFrame(Statistique)

    def separation(self,req_ww,req_crm):
        distination_ww  = self.do_separation(req_ww, "request_destination", "country", "/")
        distination_crm = self.do_separation(req_crm, "destination_lib_francais", "pays", "/")
        distination_crm = self.do_separation(distination_crm, "destination", "country", ",")
        destination     = distination_crm.append(distination_ww)
        destination     = destination[destination["destination"] != ""]
        destination     = destination[destination["country"] != ""]
        return destination

    def indexed_country(self,lists,root_model, name_model):
        df_lists = pd.DataFrame(lists)
        unique = list(df_lists.country.unique())
        indexed_country=[]
        for i in range(0,len(unique)):
            indexed_country.append({"label": unique[i], "index": i})
            df_lists.loc[df_lists['country'] == unique[i], ['country']] = i
        Etl_Model_Ml_Destination.writeToJSONFile(root_model, name_model, indexed_country)
        return df_lists

    def create_m_recomendation(self,api_ww, api_crm, all_destination,name_model,root_model):

        response_ww     = Etl_Model_Ml_Destination.web_service_response(api_ww)
        response_crm    = Etl_Model_Ml_Destination.web_service_response(api_crm)
        all_destination = Etl_Model_Ml_Destination.web_service_response(all_destination)

        destination = self.separation(response_ww, response_crm)

        uniqueid = destination.drop_duplicates(subset="destination", keep="first")
        all_destination2 = all_destination.drop_duplicates(subset="location_name", keep="first")
        destenation_id = Etl_Model_Ml_Destination.found_id(all_destination2, "location_name", "location_id", uniqueid, "destination")

        destenation = pd.merge(destination, pd.DataFrame(destenation_id), on='destination', how='inner')

        destination_country = self.found_country(all_destination, destenation)

        destination         = self.rank(pd.DataFrame(destination_country))

        destination         = self.indexed_country(destination,root_model,"indexed_country")

        Etl_Model_Ml_Destination.writeToJSONFile(root_model, name_model, destination.to_dict('records'))

        recommendation_distance = Recommendation_distnation(name_model,root_model)
        recommendation_distance.restart(name_model,root_model)

        print ("creation done")

@singleton
class Recommendation_distnation:

    alpha = 15

    def __init__(self,name_model,root_model):
        self.purchases_sparse, self.countrys, self.distinations, self.indexed_destination = self.saprce(name_model,root_model)
        self.distination_train, self.distination_test, self.distination_users_altered     = self.make_train()
        self.country_vecs, self.distination_vecs, self.distinations_arr, self.countrys_arr          = self.recommend_destination()

    def restart(self,name_model,root_model):
        self.purchases_sparse, self.countrys, self.distinations, self.indexed_destination = self.saprce(name_model,root_model)
        self.distination_train, self.distination_test, self.distination_users_altered     = self.make_train()
        self.country_vecs, self.distination_vecs, self.distinations_arr, self.countrys_arr= self.recommend_destination()

    def saprce(self,name_model,root_model):
        model_destination = Etl_Model_Ml_Destination.open_json(name_model, root_model)
        model_destination = pd.DataFrame(model_destination)

        indexed_destination = model_destination[["id", "destination"]].drop_duplicates()
        indexed_destination['id'] = indexed_destination.id.astype(str)

        model_destination['country'] = model_destination.country.astype(int)  # Convert to int for country
        model_destination = model_destination[['id', 'counts', 'country']]  # Get rid of unnecessary info
        grouped_purchased = model_destination.groupby(['country', 'id']).sum().reset_index()  # Group together

        countrys = list(np.sort(grouped_purchased.country.unique()))  # Get our unique country
        distinations = list(grouped_purchased.id.unique())
        quantity = list(grouped_purchased.counts)

        rows = grouped_purchased.country.astype('category', categories=countrys).cat.codes
        # Get the associated row indices
        cols = grouped_purchased.id.astype('category', categories=distinations).cat.codes
        # Get the associated column indices
        purchases_sparse = sparse.csr_matrix((quantity, (rows, cols)), shape=(len(countrys), len(distinations)))
        return purchases_sparse ,countrys ,distinations ,indexed_destination

    def make_train(self, pct_test=0.2):

        test_set = self.purchases_sparse.copy()  # Make a copy of the original set to be the test set.
        test_set[test_set != 0] = 1  # Store the test set as a binary preference matrix
        training_set = self.purchases_sparse.copy()  # Make a copy of the original data we can alter as our training set.
        nonzero_inds = training_set.nonzero()  # Find the indices in the ratings data where an interaction exists
        nonzero_pairs = list(
            zip(nonzero_inds[0], nonzero_inds[1]))  # Zip these pairs together of user,item index into list
        random.seed(0)  # Set the random seed to zero for reproducibility
        num_samples = int(
            np.ceil(pct_test * len(nonzero_pairs)))  # Round the number of samples needed to the nearest integer
        samples = random.sample(nonzero_pairs,
                                num_samples)  # Sample a random number of user-item pairs without replacement
        user_inds = [index[0] for index in samples]  # Get the user row indices
        item_inds = [index[1] for index in samples]  # Get the item column indices
        training_set[user_inds, item_inds] = 0  # Assign all of the randomly chosen user-item pairs to zero
        training_set.eliminate_zeros()  # Get rid of zeros in sparse array storage after update to save space
        return training_set, test_set, list(set(user_inds))  # Output the unique list of user rows that were altered

    def recommend_destination(self):
        countrys_arr = np.array(self.countrys)  # Array of destination IDs from the ratings matrix
        distinations_arr = np.array(self.distinations)
        user_vecs, item_vecs = implicit.alternating_least_squares((self.distination_train * self.alpha).astype('double'),
                                                                  factors=20,
                                                                  regularization=0.1,
                                                                  iterations=50)
        return  user_vecs, item_vecs, distinations_arr, countrys_arr

    def get_items_purchased(self,country_id, mf_train, country_list, destination_list, des_lookup):
        country_ind = np.where(country_list == country_id)[0][0]
        purchased_ind = mf_train[country_ind, :].nonzero()[1]
        prod_codes = destination_list[purchased_ind]
        destination = des_lookup.loc[des_lookup.id.isin(prod_codes)]
        return destination

    def rec_items(self,country_id, mf_train, country_vecs, destination_vecs, country_list, destination_list, des_lookup, nbdes=10):

        country_ind = np.where(country_list == country_id)[0][0]  # Returns the index row of our customer id
        pref_vec = mf_train[country_ind, :].toarray()  # Get the ratings from the training set ratings matrix
        pref_vec = pref_vec.reshape(-1) + 1  # Add 1 to everything, so that items not purchased yet become equal to 1
        pref_vec[pref_vec > 1] = 0  # Make everything already purchased zero
        rec_vector = country_vecs[country_ind, :].dot(destination_vecs.T)  # Get dot product of user vector and all item vectors
        # Scale this recommendation vector between 0 and 1
        min_max = MinMaxScaler()
        rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1, 1))[:, 0]
        recommend_vector = pref_vec * rec_vector_scaled
        # Items already purchased have their recommendation multiplied by zero
        dis_idx = np.argsort(recommend_vector)[::-1][:nbdes]  # Sort the indices of the items into order
        # of best recommendations
        rec_list = []  # start empty list to store items
        for index in dis_idx:
            code = destination_list[index]
            single = des_lookup[des_lookup["id"] == code]
            for index, sing in single.iterrows():
                rec_list.append([code, sing["destination"]])
        codes = [item[0] for item in rec_list]
        descriptions = [item[1] for item in rec_list]
        final_frame = pd.DataFrame({'id': codes, 'destination': descriptions})  # Create a dataframe
        return final_frame[['id', 'destination']]  # Switch order of columns around

    def list_recommended(self,country,date, root_name_in_co,nb_destionation,api_all_des):
        df_country  = Etl_Model_Ml_Destination.open_json("indexed_country", root_name_in_co)
        df_country  = pd.DataFrame(df_country)
        index_c     = req_country = df_country[df_country["label"] == country.upper()]
        if len(index_c)>0:
            print("existe")
            destination = self.get_items_purchased(int(index_c["index"]), self.distination_train, self.countrys_arr, self.distinations_arr, self.indexed_destination)
            print(len(destination))
            print(destination)
            if len(destination)< int(nb_destionation):
                print(len(destination))
                #destination_rec = self.rec_items(int(index_c["index"]),self.distination_train,self.country_vecs,self.distination_vecs, self.countrys_arr, self.distinations_arr, self.indexed_destination, 20)
                destination_distance = self.list_for_any(country, api_all_des)
                destination_distance = pd.DataFrame(destination_distance)
                destination = destination.append(destination_distance[['id', 'destination']][:int(nb_destionation)-len(destination)])
        else:
            print("not existe")
            destination = self.list_for_any(country, api_all_des)
            destination = pd.DataFrame(destination[:int(nb_destionation)])

        destination = self.scored_destnation(destination,date)
        return  destination[:int(nb_destionation)]

    def scored_destnation(self,destination,date):
        name_model ="model_score_destination.model"
        root_model="T:/PFE/Flask/Machine_learning/file_model"
        predection = Predection_distination(root_model + "/" + name_model)
        destination = destination.drop_duplicates(subset="id", keep="first")
        rank = []
        date_in = date.split("-")
        for index, request in destination.iterrows():
            predicit = predection.predict(date_in[2], date_in[1], date_in[0], request["id"])
            rank.append({"date": date_in[0]+"/"+date_in[1]+"/"+date_in[2], "score": predicit[0][0],'destination': request["destination"], 'id': request["id"]})

        return sorted(rank, key=lambda k: k["score"], reverse=True)

    def list_for_any(self,country,api_all):
        all_destination = Etl_Model_Ml_Destination.web_service_response(api_all)
        all_destination = pd.DataFrame(all_destination)
        all_destination = pd.DataFrame(all_destination[all_destination["location_type"] == "country"])
        one_dest = pd.DataFrame(all_destination[all_destination["location_type"] == "ocean"])
        all_destination.append(one_dest)
        all_destination = all_destination.drop_duplicates(subset="location_id", keep="first")
        geolocator = Nominatim()
        location = geolocator.geocode(country)

        newport_ri = (location.latitude, location.longitude)
        distance = []
        for index, distination in all_destination.iterrows():
            cleveland_oh = (distination["latitude"], distination["longitude"])
            distance.append({"id":distination["location_id"],"destination":distination["location_name"],"distance":vincenty(newport_ri, cleveland_oh).miles})
        distance = sorted(distance, key=lambda k: k["distance"], reverse=False)

        return   distance
