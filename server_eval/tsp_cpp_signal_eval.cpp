#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <unistd.h>

using namespace std;
namespace fs = std::filesystem;

struct Point { double x=0, y=0; };
struct Instance { string name, split, edge_type; double opt=0; vector<Point> p; };

struct Deadline {
    chrono::steady_clock::time_point t0;
    double timeout_s;
    explicit Deadline(double s): t0(chrono::steady_clock::now()), timeout_s(s) {}
    bool expired() const { return timeout_s > 0 && chrono::duration<double>(chrono::steady_clock::now()-t0).count() > timeout_s; }
    void check() const { if (expired()) throw runtime_error("timeout"); }
};

static string trim(string s) {
    auto notsp = [](int c){ return !std::isspace(c); };
    s.erase(s.begin(), find_if(s.begin(), s.end(), notsp));
    s.erase(find_if(s.rbegin(), s.rend(), notsp).base(), s.end());
    return s;
}
static vector<string> split_csv(const string& s) {
    vector<string> r; string cur; stringstream ss(s);
    while (getline(ss, cur, ',')) { cur=trim(cur); if(!cur.empty()) r.push_back(cur); }
    return r;
}
static bool contains(const vector<string>& v, const string& x) { return find(v.begin(), v.end(), x) != v.end(); }
static string csv_escape(const string& s) {
    if (s.find_first_of(",\n\r\"") == string::npos) return s;
    string o="\""; for(char c:s){ if(c=='\"') o += "\"\""; else o += c; } o += "\""; return o;
}

static long long edge_cost(const Instance& inst, int i, int j) {
    const auto &a=inst.p.at(i), &b=inst.p.at(j);
    double dx=a.x-b.x, dy=a.y-b.y;
    double d=sqrt(dx*dx+dy*dy);
    string t=inst.edge_type; for(char &c:t)c=(char)toupper(c);
    if(t.find("CEIL") != string::npos) return (long long)ceil(d);
    return (long long)floor(d+0.5);
}
static double tour_cost(const Instance& inst, const vector<int>& tour) {
    long double s=0; int n=(int)tour.size();
    for(int i=0;i<n;i++) s += edge_cost(inst, tour[i], tour[(i+1)%n]);
    return (double)s;
}
static void validate_tour(const vector<int>& tour, int n) {
    if((int)tour.size()!=n) throw runtime_error("invalid tour length");
    vector<char> seen(n,0);
    for(int x:tour){ if(x<0 || x>=n) throw runtime_error("tour city out of range"); if(seen[x]) throw runtime_error("duplicate city in tour"); seen[x]=1; }
}
static Instance read_tsp(const fs::path& pth, const string& name, const string& split, double opt) {
    ifstream f(pth); if(!f) throw runtime_error("cannot open tsp: "+pth.string());
    string line, edge="EUC_2D"; int dim=-1; bool coords=false; vector<Point> pts;
    while(getline(f,line)){
        line=trim(line); if(line.empty()) continue; string u=line; for(char &c:u)c=(char)toupper(c);
        if(u.rfind("EDGE_WEIGHT_TYPE",0)==0){ auto pos=line.find(':'); edge=trim(pos==string::npos?line.substr(16):line.substr(pos+1)); }
        if(u.rfind("DIMENSION",0)==0){ auto pos=line.find(':'); dim=stoi(trim(pos==string::npos?line.substr(9):line.substr(pos+1))); pts.reserve(dim); }
        if(u=="NODE_COORD_SECTION"){ coords=true; continue; }
        if(u=="EOF") break;
        if(coords){ stringstream ss(line); int id; double x,y; if(ss>>id>>x>>y) pts.push_back({x,y}); }
    }
    if(pts.empty()) throw runtime_error("no coordinates read: "+pth.string());
    return {name,split,edge,opt,move(pts)};
}
struct OptRow { string instance, split; double opt; };
static vector<OptRow> read_opt_csv(const fs::path& p){
    ifstream f(p); if(!f) throw runtime_error("cannot open opt csv: "+p.string());
    string line; getline(f,line); vector<OptRow> rows;
    while(getline(f,line)){ if(trim(line).empty()) continue; vector<string> c; string cur; stringstream ss(line); while(getline(ss,cur,',')) c.push_back(trim(cur)); if(c.size()>=3) rows.push_back({c[0],c[1],stod(c[2])}); }
    return rows;
}
static fs::path find_tsp(const fs::path& root, const string& name){
    vector<fs::path> c={root/(name+".tsp"), root/(name+".TSP"), root/name/(name+".tsp"), root/name/(name+".TSP")};
    for(auto &p:c) if(fs::exists(p)) return p;
    throw runtime_error("missing tsp for "+name);
}

struct RNG { mt19937_64 gen; explicit RNG(uint64_t seed): gen(seed) {} int randint(int n){ uniform_int_distribution<int>d(0,n-1); return d(gen);} int randint_range(int lo,int hi){ uniform_int_distribution<int>d(lo,hi-1); return d(gen);} };

using CandidateMap = vector<vector<int>>;
using PriorRows = vector<unordered_map<int,double>>;

static fs::path first_existing(const vector<fs::path>& c){ for(auto&p:c) if(fs::exists(p)) return p; return fs::path(); }

static CandidateMap normalize_candidates(vector<vector<int>> raw, const Instance& inst, int max_k){
    int n=(int)inst.p.size(); vector<vector<int>> tmp(n);
    for(int i=0;i<n;i++){
        unordered_set<int> s;
        for(int j: raw[i]) if(j>=0 && j<n && j!=i) s.insert(j);
        tmp[i].assign(s.begin(), s.end());
    }
    // Make bidirectional, matching the Python evaluator's normalize_candidates behavior.
    for(int i=0;i<n;i++){
        for(int j: vector<int>(tmp[i].begin(), tmp[i].end())){
            if(j>=0 && j<n && j!=i) tmp[j].push_back(i);
        }
    }
    CandidateMap out(n);
    for(int i=0;i<n;i++){
        unordered_set<int> s;
        for(int j: tmp[i]) if(j>=0 && j<n && j!=i) s.insert(j);
        out[i].assign(s.begin(), s.end());
        sort(out[i].begin(), out[i].end(), [&](int a,int b){ long long da=edge_cost(inst,i,a), db=edge_cost(inst,i,b); if(da==db) return a<b; return da<db; });
        if((int)out[i].size()>max_k) out[i].resize(max_k);
    }
    return out;
}

static CandidateMap read_candidates(const fs::path& candidate_root, const Instance& inst, int max_k){
    string name=inst.name;
    fs::path p = first_existing({
        candidate_root/(name+"_cand-popmusic-k20-s14-sol20-nn5-tr1.cand"),
        candidate_root/(name+".cand"), candidate_root/(name+".candidates"),
        candidate_root/(name+"_candidates.txt"), candidate_root/(name+".txt")
    });
    if(p.empty()) throw runtime_error("missing candidate file for "+name+" under "+candidate_root.string());
    ifstream f(p); if(!f) throw runtime_error("cannot open candidate file: "+p.string());
    vector<string> lines; string line; bool has_lkh=false;
    while(getline(f,line)){ string u=line; for(char &c:u)c=(char)toupper(c); if(u.find("CANDIDATE_SET_SECTION")!=string::npos) has_lkh=true; lines.push_back(line); }
    vector<vector<int>> rows; int max_node=-1; bool in_lkh=!has_lkh;
    for(string raw: lines){
        string s=trim(raw); string u=s; for(char &c:u)c=(char)toupper(c);
        if(s.empty() || s[0]=='#') continue;
        if(u.find("CANDIDATE_SET_SECTION")!=string::npos){ in_lkh=true; continue; }
        if(u.rfind("EOF",0)==0 || s=="-1"){ if(has_lkh) break; else continue; }
        if(has_lkh && !in_lkh) continue;
        vector<int> parts; string tok; stringstream ss(s);
        while(ss>>tok){ try{ parts.push_back(stoi(tok)); }catch(...){ } }
        if(parts.size()>=2){ rows.push_back(parts); for(int x:parts) max_node=max(max_node,x); }
    }
    int n=(int)inst.p.size(); int min_first=INT32_MAX, max_first=-1;
    for(auto&r:rows){ min_first=min(min_first,r[0]); max_first=max(max_first,r[0]); }
    bool one_based = has_lkh || (min_first>=1 && max_first<=n);
    vector<vector<int>> raw(n);
    for(auto &parts: rows){
        int i_raw=parts[0]; vector<int> neighs_raw;
        if(has_lkh && parts.size()>=4){ int deg=max(0,parts[1]); for(int k=0;k<deg && 2+2*k<(int)parts.size();k++) neighs_raw.push_back(parts[2+2*k]); if(deg==0){ for(int k=2;k<(int)parts.size();k+=2) neighs_raw.push_back(parts[k]); } }
        else { for(size_t k=1;k<parts.size();k++) neighs_raw.push_back(parts[k]); }
        int i=one_based?i_raw-1:i_raw; if(i<0 || i>=n) continue;
        for(int jr:neighs_raw){ int j=one_based?jr-1:jr; if(j>=0 && j<n && j!=i) raw[i].push_back(j); }
    }
    return normalize_candidates(move(raw), inst, max_k);
}

static PriorRows read_prior_rows(const fs::path& prior_txt_root, const Instance& inst){
    string name=inst.name;
    fs::path p=first_existing({
        prior_txt_root/(name+"_popmusic_edge_prior_runs30_topk5.prior.txt"),
        prior_txt_root/(name+"_edge_prior.prior.txt"),
        prior_txt_root/(name+".prior.txt")
    });
    if(p.empty()) throw runtime_error("missing prior txt for "+name+" under "+prior_txt_root.string());
    int n=(int)inst.p.size(); PriorRows rows(n); ifstream f(p); if(!f) throw runtime_error("cannot open prior txt: "+p.string());
    string line; while(getline(f,line)){
        line=trim(line); if(line.empty() || line[0]=='#') continue;
        stringstream ss(line); int a,b; double w; if(!(ss>>a>>b>>w)) continue;
        if(a>=0 && a<n && b>=0 && b<n && a!=b){ rows[a][b]=max(rows[a][b], w); rows[b][a]=max(rows[b][a], w); }
    }
    return rows;
}

static double prior_val(const PriorRows& pr, int i, int j){ auto it=pr[i].find(j); return it==pr[i].end()?0.0:it->second; }

static vector<int> nearest_neighbor_from_candidates(const Instance& inst, const CandidateMap& cand, int start, const Deadline& dl){
    int n=(int)inst.p.size(); vector<int> tour; tour.reserve(n); vector<char> vis(n,0);
    int current=start; tour.push_back(current); vis[current]=1;
    while((int)tour.size()<n){
        dl.check(); int best=-1; long long bestd=numeric_limits<long long>::max();
        for(int nb: cand[current]) if(!vis[nb]){ long long d=edge_cost(inst,current,nb); if(d<bestd){bestd=d; best=nb;} }
        if(best<0){
            for(int city=0;city<n;city++) if(!vis[city]){ if((city&16383)==0) dl.check(); long long d=edge_cost(inst,current,city); if(d<bestd){bestd=d; best=city;} }
        }
        if(best<0) throw runtime_error("candidate nearest-neighbor failed");
        tour.push_back(best); vis[best]=1; current=best;
    }
    return tour;
}

static vector<int> H_C1_candidate_nn(const Instance& inst, const CandidateMap& cand, RNG& rng, const Deadline& dl){
    // Direct C++ translation of actual LLM log candidate:
    // TM/llm-tsp-runs/tsp_llamea_popmusic_train_20260520_105250/codes/iter_001_d854c319152c7853.py
    // Mechanism: start from a random city; repeatedly choose the closest unvisited candidate-list
    // neighbor; if no candidate-list neighbor remains, choose a random unvisited city.
    int n=(int)inst.p.size();
    int current_city = rng.randint(n);
    vector<int> tour; tour.reserve(n);
    vector<char> visited(n,0);
    tour.push_back(current_city);
    visited[current_city]=1;
    for(int step=0; step<n-1; ++step){
        dl.check();
        int next_city = -1;
        long long best_cost = numeric_limits<long long>::max();
        for(int neighbor: cand[current_city]){
            if(!visited[neighbor]){
                long long c = edge_cost(inst, current_city, neighbor);
                if(c < best_cost){ best_cost = c; next_city = neighbor; }
            }
        }
        if(next_city < 0){
            vector<int> unvisited_cities;
            unvisited_cities.reserve(n - (int)tour.size());
            for(int city=0; city<n; ++city){
                if((city&16383)==0) dl.check();
                if(!visited[city]) unvisited_cities.push_back(city);
            }
            if(unvisited_cities.empty()) throw runtime_error("C1 candidate NN random fallback failed");
            next_city = unvisited_cities[rng.randint((int)unvisited_cities.size())];
        }
        tour.push_back(next_city);
        visited[next_city]=1;
        current_city=next_city;
    }
    return tour;
}

static vector<int> H_C1a_candidate_cleanup(const Instance& inst, const CandidateMap& cand, RNG& rng, const Deadline& dl){
    int n=(int)inst.p.size();
    vector<int> tour = nearest_neighbor_from_candidates(inst, cand, rng.randint(n), dl);
    // Random segment-reversal attempts.
    for(int a=0;a<n;a++){
        dl.check(); int i=rng.randint(n-1), j=rng.randint(n-1); if(i>j) swap(i,j); if(i==j) continue;
        int jp1 = (j+1<n)?j+1:0;
        long long edge1=edge_cost(inst,tour[i],tour[i+1]);
        long long edge2=edge_cost(inst,tour[j],tour[jp1]);
        long long edge3=edge_cost(inst,tour[i],tour[jp1]);
        long long edge4=edge_cost(inst,tour[j],tour[i+1]);
        if(edge3+edge4 < edge1+edge2) reverse(tour.begin()+i+1, tour.begin()+j+1);
    }
    // Adaptive 2-opt-like cleanup, direct translation of the selected Python method.
    bool improved=true; int max_iterations=n/2, iteration=0;
    while(improved && iteration<max_iterations){
        dl.check(); improved=false;
        for(int i=0;i<n-1;i++){
            if((i&15)==0) dl.check();
            for(int j=i+2;j<n;j++){
                long long edge1=edge_cost(inst,tour[i],tour[i+1]);
                long long edge2=edge_cost(inst,tour[j-1],tour[j]);
                long long edge3=edge_cost(inst,tour[i],tour[j-1]);
                long long edge4=edge_cost(inst,tour[j],tour[i+1]);
                if(edge3+edge4 < edge1+edge2){ reverse(tour.begin()+i+1, tour.begin()+j); improved=true; }
            }
        }
        iteration++;
    }
    // The original Python neighborhood-exploration loop never changes a complete valid tour
    // because every candidate neighbor is already in `tour`. We intentionally preserve that behavior.
    return tour;
}

static vector<int> H_P1_quality_prior(const Instance& inst, const PriorRows& pr, RNG& rng, const Deadline& dl){
    int n=(int)inst.p.size(); double cx=0,cy=0; for(auto&p:inst.p){cx+=p.x;cy+=p.y;} cx/=n; cy/=n;
    int current=0; double bestc=numeric_limits<double>::infinity();
    for(int i=0;i<n;i++){ double d=hypot(inst.p[i].x-cx, inst.p[i].y-cy); if(d<bestc){bestc=d;current=i;} }
    vector<int> tour; tour.reserve(n); vector<char> vis(n,0); tour.push_back(current); vis[current]=1;
    while((int)tour.size()<n){
        dl.check(); double max_prior=-1.0; long long max_edge_cost=numeric_limits<long long>::max(); int next=-1;
        for(int i=0;i<n;i++) if(!vis[i]){
            if((i&16383)==0) dl.check();
            double pv=prior_val(pr,current,i); long long cost=edge_cost(inst,current,i);
            if(pv>max_prior || (pv==max_prior && cost<max_edge_cost)){ max_prior=pv; next=i; max_edge_cost=cost; }
        }
        if(next<0) throw runtime_error("P1 prior construction failed");
        tour.push_back(next); vis[next]=1; current=next;
    }
    for(int a=0;a<10;a++){
        dl.check(); int idx1=rng.randint_range(0,n-1), idx2=rng.randint_range(0,n-1); (void)idx2;
        vector<int> nt; nt.reserve(n); nt.insert(nt.end(), tour.begin(), tour.begin()+idx1); nt.insert(nt.end(), tour.rbegin(), tour.rend()-idx1);
        if(tour_cost(inst,nt) < tour_cost(inst,tour)) tour.swap(nt);
    }
    return tour;
}

static vector<int> H_P2_prior_dominant(const Instance& inst, const PriorRows& pr, int start, const Deadline& dl){
    int n=(int)inst.p.size(); vector<int> tour; tour.reserve(n); vector<char> vis(n,0);
    int cur=start; tour.push_back(cur); vis[cur]=1;
    while((int)tour.size()<n){
        dl.check(); double max_score=-numeric_limits<double>::infinity(); int next=-1;
        for(int nb=0; nb<n; nb++) if(!vis[nb]){
            if((nb&16383)==0) dl.check();
            double score=prior_val(pr,cur,nb);
            if(next<0 || score>max_score || (score==max_score && edge_cost(inst,cur,nb) < edge_cost(inst,cur,next))){ max_score=score; next=nb; }
        }
        if(next<0) throw runtime_error("P2 prior-dominant construction failed");
        tour.push_back(next); vis[next]=1; cur=next;
    }
    return tour;
}

static vector<int> H_P3_fast_prior_lookahead(const Instance& inst, const PriorRows& pr, int start, const Deadline& dl){
    int n=(int)inst.p.size(); vector<int> tour; tour.reserve(n); vector<char> vis(n,0);
    int cur=start; tour.push_back(cur); vis[cur]=1;
    while((int)tour.size()<n){
        dl.check(); double max_score=-numeric_limits<double>::infinity(); int next=-1;
        for(auto &kv: pr[cur]){ int nb=kv.first; double score=kv.second; if(!vis[nb] && (next<0 || score>max_score)){ max_score=score; next=nb; } }
        if(next<0){ long long md=numeric_limits<long long>::max(); for(int nb=0;nb<n;nb++) if(!vis[nb]){ if((nb&16383)==0) dl.check(); long long d=edge_cost(inst,cur,nb); if(d<md){md=d; next=nb;} } }
        if(next<0) throw runtime_error("P3 prior lookahead construction failed");
        tour.push_back(next); vis[next]=1; cur=next;
    }
    return tour;
}

static vector<int> method_tour(const string& method, const Instance& inst, const CandidateMap* cand, const PriorRows* pr, uint64_t seed, double timeout_s){
    Deadline dl(timeout_s); RNG rng(seed); int n=(int)inst.p.size();
    if(method=="C1_candidate_nn_constructive") { if(!cand) throw runtime_error("C1 requires candidates"); return H_C1_candidate_nn(inst,*cand,rng,dl); }
    if(method=="C1a_candidate_cleanup") { if(!cand) throw runtime_error("C1a requires candidates"); return H_C1a_candidate_cleanup(inst,*cand,rng,dl); }
    if(method=="P1_quality_prior") { if(!pr) throw runtime_error("P1 requires prior"); return H_P1_quality_prior(inst,*pr,rng,dl); }
    if(method=="P2_prior_dominant") { if(!pr) throw runtime_error("P2 requires prior"); return H_P2_prior_dominant(inst,*pr,(int)(seed%n),dl); }
    if(method=="P3_fast_prior_lookahead") { if(!pr) throw runtime_error("P3 requires prior"); return H_P3_fast_prior_lookahead(inst,*pr,(int)(seed%n),dl); }
    throw runtime_error("unknown method: "+method);
}

int main(int argc, char** argv){
    unordered_map<string,string> arg; for(int i=1;i<argc;i++){ string a=argv[i]; if(a.rfind("--",0)==0 && i+1<argc) arg[a.substr(2)]=argv[++i]; }
    string job=arg["job"], kind=arg["kind"], method=arg["method"], signal=arg.count("signal")?arg["signal"]:"signal";
    fs::path inst_root=arg["instance-root"], opt_csv=arg["optima-csv"], out_dir=arg["out-dir"];
    fs::path cand_root=arg.count("candidate-root")?fs::path(arg["candidate-root"]):fs::path();
    fs::path prior_root=arg.count("prior-txt-root")?fs::path(arg["prior-txt-root"]):fs::path();
    int reps=stoi(arg.count("reps")?arg["reps"]:"50"); uint64_t global_seed=stoull(arg.count("global-seed")?arg["global-seed"]:"12345"); double timeout_s=stod(arg.count("timeout-s")?arg["timeout-s"]:"900");
    int max_candidates=stoi(arg.count("max-candidates")?arg["max-candidates"]:"20"); vector<string> wanted=split_csv(arg["instances"]); fs::create_directories(out_dir);
    vector<OptRow> opt=read_opt_csv(opt_csv); vector<Instance> instances;
    for(auto&r:opt){ if(!wanted.empty() && !contains(wanted,r.instance)) continue; instances.push_back(read_tsp(find_tsp(inst_root,r.instance),r.instance,r.split,r.opt)); }
    fs::path raw=out_dir/"raw_results.csv"; bool exists=fs::exists(raw); ofstream csv(raw,ios::app);
    if(!exists) csv << "signal_category,heuristic_id,heuristic_label,code_path,instance_name,split,n,rep,seed,objective_value,reference_value,gap_ref_pct,runtime_s,status,error_type,error_message,candidate_edge_count,total_edges,candidate_edge_share,hostname\n";
    char host[256]; gethostname(host,sizeof(host));
    unordered_map<string,CandidateMap> cand_cache; unordered_map<string,PriorRows> prior_cache;
    for(int rep=1;rep<=reps;rep++){
        for(const auto &inst: instances){
            uint64_t seed=global_seed + rep*1000003ULL + hash<string>{}(inst.name) + hash<string>{}(method);
            auto t0=chrono::steady_clock::now(); string status="ok", et="", em=""; double obj=numeric_limits<double>::quiet_NaN(), gap=numeric_limits<double>::quiet_NaN(); int cand_edges=-1,total_edges=-1; double cand_share=numeric_limits<double>::quiet_NaN();
            try{
                CandidateMap *cand=nullptr; PriorRows *pr=nullptr;
                if(signal=="candidate_list") { if(!cand_cache.count(inst.name)) cand_cache[inst.name]=read_candidates(cand_root,inst,max_candidates); cand=&cand_cache[inst.name]; }
                if(signal=="edge_prior") { if(!prior_cache.count(inst.name)) prior_cache[inst.name]=read_prior_rows(prior_root,inst); pr=&prior_cache[inst.name]; }
                auto tour=method_tour(method,inst,cand,pr,seed,timeout_s); validate_tour(tour,(int)inst.p.size()); obj=tour_cost(inst,tour); gap=100.0*(obj-inst.opt)/inst.opt;
                if(cand){ total_edges=(int)tour.size(); cand_edges=0; for(int i=0;i<total_edges;i++){ int a=tour[i], b=tour[(i+1)%total_edges]; if(find((*cand)[a].begin(), (*cand)[a].end(), b)!=(*cand)[a].end()) cand_edges++; } cand_share=(double)cand_edges/(double)total_edges; }
            } catch(const exception& e){ string msg=e.what(); if(msg.find("timeout")!=string::npos){status="timeout";et="Timeout";} else {status="error";et="RuntimeError";} em=msg; }
            double rt=chrono::duration<double>(chrono::steady_clock::now()-t0).count();
            csv << csv_escape(signal) << "," << csv_escape(job) << "," << csv_escape(method) << ",server_eval/tsp_cpp_signal_eval.cpp," << csv_escape(inst.name) << "," << csv_escape(inst.split) << "," << inst.p.size() << "," << rep << "," << seed << ",";
            if(isfinite(obj)) csv << fixed << setprecision(6) << obj; csv << "," << fixed << setprecision(6) << inst.opt << ","; if(isfinite(gap)) csv << fixed << setprecision(9) << gap;
            csv << "," << fixed << setprecision(6) << rt << "," << status << "," << et << "," << csv_escape(em) << ",";
            if(cand_edges>=0) csv << cand_edges; csv << ","; if(total_edges>=0) csv << total_edges; csv << ","; if(isfinite(cand_share)) csv << fixed << setprecision(9) << cand_share; csv << "," << host << "\n"; csv.flush();
        }
    }
    ofstream sum(out_dir/"summary_by_heuristic.csv");
    sum << "signal_category,heuristic_id,heuristic_label,total_runs,ok_runs,error_runs,timeout_runs,runtime_note\n";
    sum << signal << "," << job << "," << method << ",see_raw_results,see_raw_results,see_raw_results,see_raw_results,cpp_signal_direct_translation\n";
    return 0;
}
