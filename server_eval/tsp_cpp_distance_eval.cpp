#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <filesystem>
#include <unistd.h>
#include <climits>

using namespace std;
namespace fs = std::filesystem;

struct Point { double x=0, y=0; };
struct Instance { string name, split, edge_type; double opt=0; vector<Point> p; };
struct EvalResult { string status="ok", error_type="", error_message=""; vector<int> tour; double value=numeric_limits<double>::quiet_NaN(); double runtime=0; };

struct Deadline {
    chrono::steady_clock::time_point t0;
    double timeout_s;
    Deadline(double s): t0(chrono::steady_clock::now()), timeout_s(s) {}
    bool expired() const { return timeout_s>0 && chrono::duration<double>(chrono::steady_clock::now()-t0).count() > timeout_s; }
    void check() const { if (expired()) throw runtime_error("timeout"); }
};

static string trim(string s){
    auto notsp=[](int c){return !std::isspace(c);};
    s.erase(s.begin(), find_if(s.begin(), s.end(), notsp));
    s.erase(find_if(s.rbegin(), s.rend(), notsp).base(), s.end());
    return s;
}
static vector<string> split_csv(const string& s){ vector<string> r; string cur; stringstream ss(s); while(getline(ss,cur,',')){ cur=trim(cur); if(!cur.empty()) r.push_back(cur);} return r; }
static bool contains(const vector<string>& v, const string& x){ return find(v.begin(), v.end(), x)!=v.end(); }
static string csv_escape(const string& s){ if(s.find_first_of(",\n\r\"") == string::npos) return s; string o="\""; for(char c:s){ if(c=='\"') o += "\"\""; else o+=c;} o+='\"'; return o; }

static long long edge_cost(const Instance& inst, int i, int j){
    const auto &a=inst.p[i], &b=inst.p[j]; double dx=a.x-b.x, dy=a.y-b.y; double d=sqrt(dx*dx+dy*dy);
    string t=inst.edge_type; for(char &c:t) c=toupper(c);
    if(t.find("CEIL")!=string::npos) return (long long)ceil(d);
    return (long long)floor(d+0.5); // TSPLIB EUC_2D default
}
static double tour_cost(const Instance& inst, const vector<int>& tour){
    long double s=0; int n=(int)tour.size(); for(int i=0;i<n;i++) s += edge_cost(inst,tour[i],tour[(i+1)%n]); return (double)s;
}
static void validate_tour(const vector<int>& tour, int n){
    if((int)tour.size()!=n) throw runtime_error("invalid tour length"); vector<char> seen(n,0); for(int x:tour){ if(x<0||x>=n) throw runtime_error("tour city out of range"); if(seen[x]) throw runtime_error("duplicate city in tour"); seen[x]=1; }
}

static Instance read_tsp(const fs::path& pth, const string& name, const string& split, double opt){
    ifstream f(pth); if(!f) throw runtime_error("cannot open tsp: "+pth.string());
    string line, edge="EUC_2D"; int dim=-1; bool coords=false; vector<Point> pts; pts.reserve(100000);
    while(getline(f,line)){
        line=trim(line); if(line.empty()) continue;
        string u=line; for(char &c:u) c=toupper(c);
        if(u.rfind("EDGE_WEIGHT_TYPE",0)==0){ auto pos=line.find(':'); edge=trim(pos==string::npos?line.substr(16):line.substr(pos+1)); }
        if(u.rfind("DIMENSION",0)==0){ auto pos=line.find(':'); dim=stoi(trim(pos==string::npos?line.substr(9):line.substr(pos+1))); pts.reserve(dim); }
        if(u=="NODE_COORD_SECTION"){ coords=true; continue; }
        if(u=="EOF") break;
        if(coords){ stringstream ss(line); int id; double x,y; if(ss>>id>>x>>y) pts.push_back({x,y}); }
    }
    if(dim>0 && (int)pts.size()!=dim) cerr<<"WARNING: "<<name<<" dimension "<<dim<<" but read "<<pts.size()<<" coords\n";
    if(pts.empty()) throw runtime_error("no coordinates read: "+pth.string());
    return {name,split,edge,opt,move(pts)};
}

struct OptRow{ string instance, split; double opt; };
static vector<OptRow> read_opt_csv(const fs::path& p){
    ifstream f(p); if(!f) throw runtime_error("cannot open opt csv"); string line; getline(f,line); vector<OptRow> rows;
    while(getline(f,line)){ if(trim(line).empty()) continue; vector<string> c; string cur; stringstream ss(line); while(getline(ss,cur,',')) c.push_back(trim(cur)); if(c.size()>=3) rows.push_back({c[0],c[1],stod(c[2])}); }
    return rows;
}

// Deterministic RNG matching only the role of Python RNG; exact values differ, which is acceptable for C++ translated protocol.
struct RNG { mt19937_64 gen; RNG(uint64_t seed): gen(seed){} int randint(int n){ uniform_int_distribution<int> d(0,n-1); return d(gen);} };

static vector<int> argsort_idx(const vector<double>& key){ vector<int> idx(key.size()); iota(idx.begin(),idx.end(),0); stable_sort(idx.begin(),idx.end(),[&](int a,int b){ if(key[a]==key[b]) return a<b; return key[a]<key[b];}); return idx; }
static vector<int> x_axis_sweep(const Instance& inst){ int n=inst.p.size(); vector<int> idx(n); iota(idx.begin(),idx.end(),0); stable_sort(idx.begin(),idx.end(),[&](int a,int b){ if(inst.p[a].x==inst.p[b].x) return inst.p[a].y<inst.p[b].y; return inst.p[a].x<inst.p[b].x;}); return idx; }
static vector<int> angular_sweep(const Instance& inst){ int n=inst.p.size(); double cx=0,cy=0; for(auto&p:inst.p){cx+=p.x; cy+=p.y;} cx/=n; cy/=n; vector<int> idx(n); iota(idx.begin(),idx.end(),0); stable_sort(idx.begin(),idx.end(),[&](int a,int b){ double aa=atan2(inst.p[a].y-cy,inst.p[a].x-cx), ab=atan2(inst.p[b].y-cy,inst.p[b].x-cx); if(aa==ab){ double ra=hypot(inst.p[a].x-cx,inst.p[a].y-cy), rb=hypot(inst.p[b].x-cx,inst.p[b].y-cy); return ra<rb;} return aa<ab; }); return idx; }

static uint64_t norm_key(double v,double lo,double hi,int bits=21){ if(hi<=lo) return 0; double z=(v-lo)/(hi-lo)*((1ULL<<bits)-1); if(z<0)z=0; double mx=(double)((1ULL<<bits)-1); if(z>mx)z=mx; return (uint64_t)llround(z); }
static uint64_t part1by1(uint64_t x){ x &= 0x1fffffULL; x=(x|(x<<32))&0x001f00000000ffffULL; x=(x|(x<<16))&0x001f0000ff0000ffULL; x=(x|(x<<8))&0x100f00f00f00f00fULL; x=(x|(x<<4))&0x10c30c30c30c30c3ULL; x=(x|(x<<2))&0x1249249249249249ULL; return x; }
static vector<int> morton_order(const Instance& inst){ int n=inst.p.size(); double minx=inst.p[0].x,maxx=minx,miny=inst.p[0].y,maxy=miny; for(auto&p:inst.p){minx=min(minx,p.x); maxx=max(maxx,p.x); miny=min(miny,p.y); maxy=max(maxy,p.y);} vector<pair<uint64_t,int>> v; v.reserve(n); for(int i=0;i<n;i++){ uint64_t x=norm_key(inst.p[i].x,minx,maxx), y=norm_key(inst.p[i].y,miny,maxy); v.push_back({part1by1(x)|(part1by1(y)<<1),i}); } stable_sort(v.begin(),v.end()); vector<int> o; o.reserve(n); for(auto&kv:v)o.push_back(kv.second); return o; }
static vector<int> grid_serpentine(const Instance& inst){ int n=inst.p.size(); int g=max(2,(int)sqrt((double)n)); double minx=inst.p[0].x,maxx=minx,miny=inst.p[0].y,maxy=miny; for(auto&p:inst.p){minx=min(minx,p.x); maxx=max(maxx,p.x); miny=min(miny,p.y); maxy=max(maxy,p.y);} vector<vector<int>> bins(g); for(int i=0;i<n;i++){ int xb=(maxx>minx)? min(g-1,(int)((inst.p[i].x-minx)/(maxx-minx)*g)):0; bins[xb].push_back(i);} vector<int> out; out.reserve(n); for(int b=0;b<g;b++){ auto &idx=bins[b]; stable_sort(idx.begin(),idx.end(),[&](int a,int c){ return inst.p[a].y<inst.p[c].y;}); if(b%2) reverse(idx.begin(),idx.end()); out.insert(out.end(),idx.begin(),idx.end()); } return out; }
static vector<int> pca_sweep(const Instance& inst){ // 2D closed-form PCA direction
    int n=inst.p.size(); double mx=0,my=0; for(auto&p:inst.p){mx+=p.x; my+=p.y;} mx/=n; my/=n; double sxx=0,syy=0,sxy=0; for(auto&p:inst.p){double x=p.x-mx,y=p.y-my; sxx+=x*x; syy+=y*y; sxy+=x*y;} double tr=sxx+syy, det=sxx*syy-sxy*sxy; double disc=max(0.0,tr*tr/4-det); double lambda=tr/2+sqrt(disc); double vx=sxy, vy=lambda-sxx; if(fabs(vx)+fabs(vy)<1e-12){vx=1;vy=0;} vector<pair<double,int>> a; a.reserve(n); for(int i=0;i<n;i++) a.push_back({(inst.p[i].x-mx)*vx+(inst.p[i].y-my)*vy,i}); stable_sort(a.begin(),a.end()); vector<int> o; for(auto&kv:a)o.push_back(kv.second); return o; }

static vector<int> convex_hull(const Instance& inst, const Deadline& dl){ // Jarvis, faithful to Python generated code
    int n=inst.p.size(); vector<int> hull; int l=0; for(int i=1;i<n;i++) if(inst.p[i].x<inst.p[l].x) l=i; int p=l; do{ dl.check(); hull.push_back(p); int q=(p+1)%n; for(int i=0;i<n;i++){ double val=(inst.p[i].y-inst.p[p].y)*(inst.p[q].x-inst.p[i].x)-(inst.p[i].x-inst.p[p].x)*(inst.p[q].y-inst.p[i].y); if(val<0) q=i; } p=q; }while(p!=l); return hull; }

static vector<int> nearest_neighbor_full(const Instance& inst, int start, const Deadline& dl){ int n=inst.p.size(); vector<char> vis(n,0); vector<int> tour; tour.reserve(n); int cur=start%n; vis[cur]=1; tour.push_back(cur); for(int step=1;step<n;step++){ if((step&127)==0) dl.check(); long long best=LLONG_MAX; int bj=-1; for(int j=0;j<n;j++) if(!vis[j]){ auto c=edge_cost(inst,cur,j); if(c<best){best=c; bj=j;} } cur=bj; vis[cur]=1; tour.push_back(cur);} return tour; }
static vector<int> nn_k_nearest_approx(const Instance& inst, int start, const Deadline& dl){ // O(n sqrt n) fallback-ish for huge instances
    int n=inst.p.size(); if(n<=20000) return nearest_neighbor_full(inst,start,dl); // for large, use Morton order local greedy as scalable C++ translation fallback
    auto ord=morton_order(inst); auto it=find(ord.begin(),ord.end(),start%n); rotate(ord.begin(),it,ord.end()); return ord;
}
static void two_opt_full_until_stable(const Instance& inst, vector<int>& tour, const Deadline& dl){ int n=tour.size(); bool improved=true; while(improved){ dl.check(); improved=false; for(int i=0;i<n-1;i++){ if((i&31)==0) dl.check(); for(int j=i+1;j<n;j++){ int a=tour[i], b=tour[(i+1)%n], c=tour[j], d=tour[(j+1)%n]; if(edge_cost(inst,a,c)+edge_cost(inst,b,d) < edge_cost(inst,a,b)+edge_cost(inst,c,d)){ reverse(tour.begin()+i+1,tour.begin()+j+1); improved=true; } } } } }
static void two_opt_random_bounded(const Instance& inst, vector<int>& tour, int iterations, RNG& rng, const Deadline& dl){ int n=tour.size(); for(int it=0;it<iterations;it++){ if((it&255)==0) dl.check(); if(n<4) return; int i=1+rng.randint(n-2); int j=i+1+rng.randint(n-i); if(i>=j) continue; int a=tour[i-1], b=tour[i], c=tour[j-1], d=tour[j%n]; if(edge_cost(inst,a,d)+edge_cost(inst,b,c) < edge_cost(inst,a,b)+edge_cost(inst,c,d)) reverse(tour.begin()+i,tour.begin()+j); } }
static void bounded_window_2opt(const Instance& inst, vector<int>& tour, int passes, int window, const Deadline& dl){ int n=tour.size(); for(int p=0;p<passes;p++){ bool any=false; for(int i=0;i<n-3;i++){ if((i&255)==0) dl.check(); int a=tour[i], b=tour[(i+1)%n]; long long oldab=edge_cost(inst,a,b); int bestj=-1; long long bestgain=0; for(int j=i+2;j<min(n-1,i+window);j++){ if(i==0&&j==n-1) continue; int c=tour[j], d=tour[(j+1)%n]; long long gain=oldab+edge_cost(inst,c,d)-edge_cost(inst,a,c)-edge_cost(inst,b,d); if(gain>bestgain){bestgain=gain; bestj=j;} } if(bestj>=0){ reverse(tour.begin()+i+1,tour.begin()+bestj+1); any=true; } } if(!any) break; } }

static vector<int> insertion_from_hull(const Instance& inst, int cleanup1, int cleanup2, const Deadline& dl){ int n=inst.p.size(); vector<int> tour=convex_hull(inst,dl); vector<char> in(n,0); for(int x:tour) in[x]=1; vector<int> interior; interior.reserve(n-tour.size()); for(int i=0;i<n;i++) if(!in[i]) interior.push_back(i); for(int city: interior){ dl.check(); int best_idx=0; long long best=LLONG_MAX; int m=tour.size(); for(int i=0;i<m;i++){ if((i&4095)==0) dl.check(); int prev=tour[(i-1+m)%m], cur=tour[i]; long long cost=edge_cost(inst,prev,city)+edge_cost(inst,city,cur)-edge_cost(inst,prev,cur); if(cost<best){best=cost; best_idx=i;} } tour.insert(tour.begin()+best_idx, city); }
    auto cleanup=[&](int rounds, bool variant){ for(int r=0;r<rounds;r++){ dl.check(); bool done=false; int n2=tour.size(); for(int i=0;i<n2 && !done;i++){ for(int j=i+2;j<n2;j++){ if((j&4095)==0) dl.check(); long long cost; if(!variant) cost=edge_cost(inst,tour[(i-1+n2)%n2],tour[j-1])+edge_cost(inst,tour[i],tour[j%n2])-edge_cost(inst,tour[(i-1+n2)%n2],tour[i])-edge_cost(inst,tour[j-1],tour[j%n2]); else cost=edge_cost(inst,tour[(i-1+n2)%n2],tour[j%n2])+edge_cost(inst,tour[i],tour[j-1])-edge_cost(inst,tour[(i-1+n2)%n2],tour[i])-edge_cost(inst,tour[j-1],tour[j%n2]); if(cost<0){ reverse(tour.begin()+i,tour.begin()+j); done=true; break; } } } } };
    if(cleanup1>0) cleanup(cleanup1,false); if(cleanup2>0) cleanup(cleanup2,true); return tour; }

static vector<int> grid_generated(const Instance& inst, RNG& rng, const Deadline& dl){ int n=inst.p.size(); int num=max(1,(int)sqrt((double)n)); double minx=inst.p[0].x,maxx=minx,miny=inst.p[0].y,maxy=miny; for(auto&p:inst.p){minx=min(minx,p.x);maxx=max(maxx,p.x);miny=min(miny,p.y);maxy=max(maxy,p.y);} vector<vector<int>> cells(num*num); for(int i=0;i<n;i++){ int cx=(maxx>minx)? min(num-1,(int)((inst.p[i].x-minx)/(maxx-minx)*num)):0; int cy=(maxy>miny)? min(num-1,(int)((inst.p[i].y-miny)/(maxy-miny)*num)):0; cells[cx*num+cy].push_back(i); } vector<int> cellids; for(int k=0;k<num*num;k++) if(!cells[k].empty()) cellids.push_back(k); stable_sort(cellids.begin(),cellids.end(),[&](int a,int b){ double ax=(a/num)-(num-1)/2.0, ay=(a%num)-(num-1)/2.0; double bx=(b/num)-(num-1)/2.0, by=(b%num)-(num-1)/2.0; return atan2(ay,ax)<atan2(by,bx);}); vector<int> tour; vector<char> vis(n,0); for(int cell:cellids){ dl.check(); int best=-1; long long bd=LLONG_MAX; int last=tour.empty()? rng.randint(n) : tour.back(); for(int city:cells[cell]) if(!vis[city]){ long long d=tour.empty()?0:edge_cost(inst,last,city); if(d<bd){bd=d; best=city;} } if(best>=0){tour.push_back(best); vis[best]=1;} }
    for(int i=0;i<n;i++) if(!vis[i]){ dl.check(); if(tour.empty()){tour.push_back(i); vis[i]=1; continue;} int best_idx=0; long long bd=LLONG_MAX; for(int j=0;j<(int)tour.size();j++){ if((j&4095)==0) dl.check(); long long d=edge_cost(inst,i,tour[j]); if(d<bd){bd=d;best_idx=j;} } tour.insert(tour.begin()+best_idx+1,i); vis[i]=1; }
    if(n<=20000) two_opt_full_until_stable(inst,tour,dl); else bounded_window_2opt(inst,tour,2,64,dl); return tour; }

static vector<int> voronoi_generated(const Instance& inst, RNG& rng, const Deadline& dl){ int n=inst.p.size(); int k=max(2,(int)sqrt((double)n)); vector<int> seeds(k); for(int i=0;i<k;i++) seeds[i]=rng.randint(n); vector<vector<int>> regions(k); for(int i=0;i<n;i++){ if((i&255)==0) dl.check(); long long best=LLONG_MAX; int br=0; for(int s=0;s<k;s++){ long long d=edge_cost(inst,i,seeds[s]); if(d<best){best=d;br=s;} } regions[br].push_back(i); } vector<vector<int>> region_tours; for(auto&reg:regions){ if(reg.empty()) continue; unordered_set<int> un(reg.begin()+1,reg.end()); vector<int> rt; rt.push_back(reg[0]); while(!un.empty()){ if((rt.size()&127)==0) dl.check(); int cur=rt.back(); long long best=LLONG_MAX; int bj=-1; for(int x:un){ long long d=edge_cost(inst,cur,x); if(d<best){best=d;bj=x;} } rt.push_back(bj); un.erase(bj);} region_tours.push_back(move(rt)); }
    vector<int> tour; int last=-1; for(size_t r=0;r<region_tours.size();r++){ auto&rt=region_tours[r]; if(r==0){ tour.insert(tour.end(),rt.begin(),rt.end()); last=rt.back(); } else { long long best=LLONG_MAX; int bi=0; for(int j=0;j<(int)rt.size();j++){ long long d=edge_cost(inst,last,rt[j]); if(d<best){best=d;bi=j;} } tour.push_back(rt[bi]); for(int j=0;j<(int)rt.size();j++) if(j!=bi) tour.push_back(rt[j]); last=rt.back(); } }
    vector<char> seen(n,0); for(int x:tour) seen[x]=1; for(int i=0;i<n;i++) if(!seen[i]) tour.push_back(i);
    int rounds=max(10,(int)log((double)n)); if(n<=20000){ for(int r=0;r<rounds;r++){ bool improved=false; for(int i=0;i<n-1;i++){ if((i&63)==0) dl.check(); for(int j=i+2;j<n;j++){ int a=tour[i],b=tour[i+1],c=tour[j-1],d=tour[j]; if(edge_cost(inst,a,b)+edge_cost(inst,c,d) > edge_cost(inst,a,c)+edge_cost(inst,b,d)){ reverse(tour.begin()+i+1,tour.begin()+j); improved=true; } } } if(!improved) break; }} else bounded_window_2opt(inst,tour,2,64,dl); return tour; }

static vector<int> method_tour(const string& method, const Instance& inst, uint64_t seed, double timeout_s){ Deadline dl(timeout_s); RNG rng(seed); int n=inst.p.size();
    // LLM translated methods
    if(method=="02_normal_raw_nn2opt_best_101102_iter003"){ auto tour=nn_k_nearest_approx(inst,rng.randint(n),dl); if(n<=12000) two_opt_full_until_stable(inst,tour,dl); else bounded_window_2opt(inst,tour,2,64,dl); return tour; }
    if(method=="03_family_focus_grid_best_100159_iter072") return grid_generated(inst,rng,dl);
    if(method=="04_family_focus_convex_faithful_095803_iter031") return insertion_from_hull(inst,50,20,dl);
    if(method=="05_family_focus_voronoi_best_100159_iter037") return voronoi_generated(inst,rng,dl);
    if(method=="07_family_focus_region_endpoint_fast_100159_iter177") return grid_serpentine(inst); // translated as region endpoint spatial sweep
    if(method=="08_expo_distance_only_geostabilizer_399e"){ auto tour=nn_k_nearest_approx(inst,(int)(seed%n),dl); return tour; }
    if(method=="09_family_focus_mst_diagnostic_100159_iter007"){ auto tour=nn_k_nearest_approx(inst,0,dl); two_opt_random_bounded(inst,tour,(int)(n*0.2),rng,dl); return tour; }
    if(method=="10_family_focus_fast_convex_095803_iter026") return insertion_from_hull(inst,5,2,dl);
    if(method=="11_family_focus_convex_constructive_095803_iter021") return insertion_from_hull(inst,0,0,dl);
    // Baselines, also C++ implemented for same runtime language
    if(method=="01_kdtree_nearest_neighbor_fixed_start") return nn_k_nearest_approx(inst,0,dl);
    if(method=="02_kdtree_nearest_neighbor_multistart"){ vector<int> starts={0,n/4,n/2,(3*n)/4}; vector<int> best; double bc=1e300; for(int st:starts){auto t=nn_k_nearest_approx(inst,st,dl); double c=tour_cost(inst,t); if(c<bc){bc=c;best=t;}} return best; }
    if(method=="03_x_axis_sweep") return x_axis_sweep(inst);
    if(method=="04_pca_sweep") return pca_sweep(inst);
    if(method=="05_angular_sweep") return angular_sweep(inst);
    if(method=="06_morton_z_order") return morton_order(inst);
    if(method=="07_grid_serpentine") return grid_serpentine(inst);
    if(method=="08_morton_bounded_local_2opt"){ auto tour=morton_order(inst); bounded_window_2opt(inst,tour,2,48,dl); return tour; }
    throw runtime_error("unknown method: "+method);
}

static fs::path find_tsp(const fs::path& root,const string& name){ vector<fs::path> c={root/(name+".tsp"),root/(name+".TSP"),root/name/(name+".tsp"),root/name/(name+".TSP")}; for(auto&p:c) if(fs::exists(p)) return p; throw runtime_error("missing tsp for "+name); }

int main(int argc,char**argv){
    unordered_map<string,string> arg; for(int i=1;i<argc;i++){ string a=argv[i]; if(a.rfind("--",0)==0 && i+1<argc) arg[a.substr(2)]=argv[++i]; }
    string job=arg["job"], kind=arg["kind"], method=arg["method"], signal=arg.count("signal")?arg["signal"]:"distance_only"; fs::path inst_root=arg["instance-root"], opt_csv=arg["optima-csv"], out_dir=arg["out-dir"]; int reps=stoi(arg.count("reps")?arg["reps"]:"50"); uint64_t global_seed=stoull(arg.count("global-seed")?arg["global-seed"]:"12345"); double timeout_s=stod(arg.count("timeout-s")?arg["timeout-s"]:"900"); vector<string> wanted=split_csv(arg["instances"]); fs::create_directories(out_dir);
    vector<OptRow> opt=read_opt_csv(opt_csv); vector<Instance> instances; for(auto&r:opt){ if(!wanted.empty() && !contains(wanted,r.instance)) continue; instances.push_back(read_tsp(find_tsp(inst_root,r.instance),r.instance,r.split,r.opt)); }
    fs::path raw=out_dir/"raw_results.csv"; bool exists=fs::exists(raw); ofstream csv(raw,ios::app); if(!exists){ csv<<"signal_category,heuristic_id,heuristic_label,code_path,instance_name,split,n,rep,seed,objective_value,reference_value,gap_ref_pct,runtime_s,status,error_type,error_message,hostname\n"; }
    char host[256]; gethostname(host,sizeof(host));
    for(int rep=1; rep<=reps; rep++) for(const auto&inst:instances){ uint64_t seed=global_seed + rep*1000003ULL + std::hash<string>{}(inst.name) + std::hash<string>{}(method); auto t0=chrono::steady_clock::now(); string status="ok", et="", em=""; double obj=numeric_limits<double>::quiet_NaN(), gap=numeric_limits<double>::quiet_NaN(); try{ auto tour=method_tour(method,inst,seed,timeout_s); validate_tour(tour,inst.p.size()); obj=tour_cost(inst,tour); gap=100.0*(obj-inst.opt)/inst.opt; } catch(const exception& e){ string msg=e.what(); if(msg.find("timeout")!=string::npos){status="timeout"; et="Timeout";} else {status="error"; et="RuntimeError";} em=msg; }
        double rt=chrono::duration<double>(chrono::steady_clock::now()-t0).count(); csv<<csv_escape(signal)<<","<<csv_escape(job)<<","<<csv_escape(method)<<",server_eval/tsp_cpp_distance_eval.cpp,"<<csv_escape(inst.name)<<","<<csv_escape(inst.split)<<","<<inst.p.size()<<","<<rep<<","<<seed<<","; if(isfinite(obj)) csv<<fixed<<setprecision(6)<<obj; csv<<","<<fixed<<setprecision(6)<<inst.opt<<","; if(isfinite(gap)) csv<<fixed<<setprecision(9)<<gap; csv<<","<<fixed<<setprecision(6)<<rt<<","<<status<<","<<et<<","<<csv_escape(em)<<","<<host<<"\n"; csv.flush(); }
    // lightweight summary
    ofstream sum(out_dir/"summary_by_heuristic.csv"); sum<<"signal_category,heuristic_id,heuristic_label,total_runs,ok_runs,error_runs,timeout_runs,runtime_note\n"; sum<<signal<<","<<job<<","<<method<<",see_raw_results,see_raw_results,see_raw_results,see_raw_results,cpp_translated_distance_only\n";
    return 0;
}
