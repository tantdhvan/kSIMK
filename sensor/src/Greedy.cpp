#include "Greedy.h"
#include <iostream>
#include <math.h>
#include <unordered_map>
#include <algorithm>

Greedy::Greedy(Network *g) : Framework(g), no_queries(0) {}

Greedy::~Greedy() {}

//greedy
double Greedy::get_solution(bool is_ds)
{
    kseeds seedsf;
    int no_nodes = g->get_no_nodes();
    no_queries = 0;
    vector<bool> v(no_nodes, false);
    double C_S[] = {0.0, 0.0, 0.0};
    double current_f_s = 0;
    while (true)
    {
        int i_max = -1, e_max = -1;
        double delta = 0, f_max = 0;
        double sumcost=0;
        for (int e = 0; e < no_nodes; e++)
        {
            sumcost+=cost_matrix[e];
            //cout<<"cost_matrix[e]: "<<cost_matrix[e]<<endl;
            if (v[e] == true) continue;
            #pragma omp parallel for
            for (int i = 0; i < Constants::K; i++)
            {
                if (C_S[i] + cost_matrix[e] > Constants::BUDGET) continue;
                kseeds tmp_seedsf = seedsf;
                tmp_seedsf.push_back(kpoint(e, i));
                double current_max_f = estimate_influence(tmp_seedsf);
                cout<<"tmp_seedsf: "<<current_max_f<<endl;
                no_queries++;
                double tmp_delta = (current_max_f - current_f_s) / cost_matrix[e];
                if (tmp_delta > delta)
                {
                    #pragma omp critical
                    {
                        e_max = e;
                        i_max = i;
                        delta = tmp_delta;
                        f_max = current_max_f;
                    }
                }
            }
            //cout<<"f_max: "<<f_max<<endl;
        }
        //cout<<"sumcost: "<<sumcost<<endl;
        if (i_max == -1 || e_max == -1) break;
        seedsf.push_back(kpoint(e_max, i_max));
        C_S[i_max] += cost_matrix[e_max];
        current_f_s = f_max;
        v[e_max] = true;
        cout<<"C_S: "<<C_S[0]<<" "<<C_S[1]<<" "<<C_S[2]<<" current_f_s: "<<f_max<<endl;
    }
    //cout<<"seedsf size: "<<seedsf.size()<<endl;
    cout<<"C_S: "<<C_S[0]<<" "<<C_S[1]<<" "<<C_S[2]<<endl;
    return current_f_s;
}
//thuật toán chính
double Greedy::get_solution1(bool is_ds)
{
    //cout<<"start sol"<<endl;
    int no_nodes = g->get_no_nodes();
    double alpha = 0.25;
    vector<myType> nguong;
    vector<int> nguongj;
    int e_max=-1, i_max=-1;
    double f_e_i_max=-1;
    size_t count=0;
    for (int e = 0; e < no_nodes; e++)
    {
        std::unordered_map<string, pair<int,double>> myMap;
        int i_m =-1;
        double f_i=-1;
        #pragma omp parallel for
        for (int i = 0; i < Constants::K; i++)
        {
            kseeds tmp_seeds1;
            tmp_seeds1.push_back(kpoint(e, i));
            double tmp_f = estimate_influence(tmp_seeds1);
            no_queries++;
            
            if(tmp_f > f_i)
            {
                #pragma omp critical
                {
                    i_m = i;
                    f_i = tmp_f;
                }
            }
        }
        myMap[""] = pair<int,double>(i_m,f_i);
        if(f_i > f_e_i_max)
        {
            e_max = e;
            i_max = i_m;
            f_e_i_max = f_i;
            int j=0;
            while (true)
            {
                double tmp = pow((1 + Constants::EPS), j);
                if (tmp > f_e_i_max * Constants::BUDGET*Constants::K) break;
                if (tmp >= f_e_i_max)
                {
                    auto it = std::find(nguongj.begin(), nguongj.end(), j);
                    if (it == nguongj.end())
                    {
                        myType tmp_ng;
                        tmp_ng.nguong=j;
                        tmp_ng.nguongd=tmp;
                        tmp_ng.cost={0.0, 0.0, 0.0};
                        tmp_ng.check = vector<bool>(no_nodes, false);
                        tmp_ng.code="";
                        nguong.push_back(tmp_ng);
                        nguongj.push_back(j);
                    }
                }
                j++;
            }
        }
        
        //#pragma omp parallel for
        for (int t = 0; t < nguong.size(); t++)
        {
            myType& nguongt = nguong[t];
            if(nguongt.nguongd<f_e_i_max) continue;
            //size_t hashValue = nguongt.id;
            //cout<<"----- nguongt.s:"<<nguongt.s.size()<<" hashValue: " << hashValue<<endl;
            auto it =myMap.find(nguongt.code);
            double f=0;
            int iv_max = -1;
            if (it == myMap.end())
            {
                //cout<<"----- founded"<<endl;
                #pragma omp parallel for
                for (int i = 0; i < Constants::K; i++)
                {
                    if (nguongt.cost[i] + cost_matrix[e] > Constants::BUDGET) continue;

                    kseeds tmp= nguongt.s;
                    tmp.push_back(kpoint(e, i));
                    double max_fstnow = estimate_influence(nguongt.s);
                    no_queries++;

                    if (max_fstnow > f)
                    {
                        #pragma opm critical
                        {
                            iv_max = i;
                            f = max_fstnow;
                        }
                    }
                }
                myMap[nguongt.code] = {iv_max, f};
            }else
            {
                count++;
                iv_max =myMap[nguongt.code].first;
                f=myMap[nguongt.code].second;
            }
            //cout<<"f:"<<f<<endl;
            if (iv_max != -1)
            {
                if ((f - nguongt.current_f) >= cost_matrix[e] * alpha * nguongt.nguongd / Constants::BUDGET)
                {
                    nguongt.s.push_back(kpoint(e, iv_max));
                    nguongt.current_f = f;
                    nguongt.cost[iv_max] += cost_matrix[e];
                    nguongt.check[e]=true;
                    nguongt.code+="1";
                    //cout<<"them"<<endl;
                }else
                {
                    nguongt.code+="0";
                }
            }
        }
    }
    cout<<"count: "<<count<<endl;
    cout<<"no_queries: "<<no_queries<<endl;
    
    for (int e = 0; e < no_nodes; e++)
    {
        std::unordered_map<string, pair<int,double>> myMap;
        for (int t = 0; t < nguong.size(); t++)
        {
            myType& nguongt = nguong[t];
            if(nguongt.nguongd<f_e_i_max) continue;
            if(nguongt.check[e]==true) continue;

            auto it =myMap.find(nguongt.code);

            int imax=-1;
            double f=0;
            if(it != myMap.end())
            {
                imax = myMap[nguongt.code].first;
                f = myMap[nguongt.code].second;
            }
            else
            {
                for (int i = 0; i < Constants::K; i++)
                {
                    if(nguongt.cost[i] + cost_matrix[e] > Constants::BUDGET) continue;
                    
                    nguongt.s.push_back(kpoint(e, i));
                    double max_fstnow = estimate_influence(nguongt.s);
                    no_queries++;
                    nguongt.s.pop_back();

                    if (max_fstnow > f && max_fstnow>nguongt.current_f)
                    {
                        imax = i;
                        f = max_fstnow;
                    }
                }
            }
            if(imax!=-1)
            {
                nguongt.s.push_back(kpoint(e, imax));
                nguongt.current_f = f;
                nguongt.cost[imax] += cost_matrix[e];
                nguongt.check[e]=true;
                nguongt.code+="1";
            }else
            {
                nguongt.code+="0";
            }
        }
    }
    cout<<"count: "<<count<<endl;
    cout<<"no_queries: "<<no_queries<<endl;
    double result=0;
    for(int i=0;i<nguong.size();i++)
    {
        cout<<"nguong[i].current_f: "<<nguong[i].current_f<<endl;
        if(nguong[i].current_f>result) result=nguong[i].current_f;
    }
    return result;
}
int Greedy::get_no_queries() { return no_queries; }