#include <array>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
using namespace std;
struct Neuron{double a,b,c,d;};
struct Sample{double x,y,rx,ry;};
using Mat=array<double,4>;
#ifndef SWING_H
#define SWING_H 10000000
#endif
const size_t h=SWING_H; const int N=2000;
vector<Neuron> state(h),tmp(h),acc(h); vector<array<double,2>> initial(h);
vector<Sample> training(N); vector<size_t> uncertain[2];
#ifndef PROBE_THETA
#define PROBE_THETA 4e-5
#endif
double rmin[2],rmax[2],A0,theta=PROBE_THETA,ep=sqrt(.04/h);
ofstream simlog; double qtheory=.0005336215483052569, ztheory=.0001315793049507211,ltheory=.0001020610673944922;
int status(const Neuron &v,int g){
 double center=g?v.b:v.a, coeff=g?v.a:v.b;
 double x=center+coeff*rmin[g],y=center+coeff*rmax[g];
 if(min(x,y)>0)return 1;if(max(x,y)<0)return -1;return 0;
}
void addmat(Mat &m,const Neuron &v){m[0]+=v.c*v.a;m[1]+=v.c*v.b;m[2]+=v.d*v.a;m[3]+=v.d*v.b;}
void stage(vector<Neuron>& src, vector<Neuron>& dst,double basecoef,double acccoef,int mode,double t,bool report){
 Mat bulk[2]{};uncertain[0].clear();uncertain[1].clear();
 double f1=0,f2=0,S=0,uu=0,mm=0,RR=0,pp=0,coremass=0,tanU=0,tanW=0;
 double uv=0,cdiff=0,bdiff=0,ddiff=0,weak1=0,weak2=0;
 double maxJD=0,maxJB=0,maxJP=0,coregamma=0,maxv=0,mincorea=1e100;
 double At=1/sqrt(1+(1/(A0*A0)-1)*exp(-t));
 for(size_t i=0;i<h;i++){
  const auto &v=src[i];for(int g=0;g<2;g++){int k=status(v,g);if(k>0)addmat(bulk[g],v);else if(!k)uncertain[g].push_back(i);}
  if(report){
   double qp=max(0.,v.a*sin(theta)+v.b*cos(theta));f1+=v.c*qp;f2+=v.d*qp;
   double ap=max(0.,v.a),bi=max(0.,v.b),p=v.c-v.a;
   S+=v.a*v.a+v.b*v.b;uu+=ap*ap;mm+=v.c*ap;RR+=v.d*ap;pp+=p*p;
   weak1+=v.c*bi;weak2+=v.d*bi;
   double vi=max(0.,initial[i][0])/A0;uv+=ap*vi;
   double cc=v.c-(At*vi+min(0.,initial[i][0]));cdiff+=cc*cc;
   bdiff+=pow(v.b-initial[i][1],2);ddiff+=pow(v.d-initial[i][1],2);
   maxJD=max(maxJD,abs(v.d-initial[i][1])/ep);maxJB=max(maxJB,abs(v.b-initial[i][1])/ep);maxJP=max(maxJP,abs(v.c-v.a)/ep);maxv=max(maxv,hypot(v.a,v.b)/(ep*At/A0));
   if(initial[i][0]>=abs(initial[i][1])){coremass+=v.a*v.c;coregamma=max(coregamma,1-v.a/(At*initial[i][0]/A0));mincorea=min(mincorea,v.a);tanU=max(tanU,abs(v.b/v.a));tanW=max(tanW,abs(v.d/v.c));}
  }
 }
 Mat grad[2]{};double loss=0;
 for(int j=0;j<N;j++){
  int g=j>=N/2;auto &x=training[j];double out1=bulk[g][0]*x.x+bulk[g][1]*x.y,out2=bulk[g][2]*x.x+bulk[g][3]*x.y;
  for(size_t i:uncertain[g]){const auto &v=src[i];double q=max(0.,v.a*x.x+v.b*x.y);out1+=v.c*q;out2+=v.d*q;}
  x.rx=x.x-out1;x.ry=x.y-out2;
  grad[g][0]+=x.rx*x.x/N;grad[g][1]+=x.rx*x.y/N;grad[g][2]+=x.ry*x.x/N;grad[g][3]+=x.ry*x.y/N;
  loss+=(x.rx*x.rx+x.ry*x.ry)/(2*N);
 }
 double fd1=0,fd2=0,maxbalance=0;
 for(size_t i=0;i<h;i++){
  Neuron v=src[i];Mat G{};
  for(int g=0;g<2;g++){
   int k=status(v,g);if(k>0){for(int j=0;j<4;j++)G[j]+=grad[g][j];}
   else if(!k){for(int j=g*N/2;j<(g+1)*N/2;j++){const auto &x=training[j];if(v.a*x.x+v.b*x.y>0){G[0]+=x.rx*x.x/N;G[1]+=x.rx*x.y/N;G[2]+=x.ry*x.x/N;G[3]+=x.ry*x.y/N;}}}
  }
  Neuron dv{G[0]*v.c+G[2]*v.d,G[1]*v.c+G[3]*v.d,G[0]*v.a+G[1]*v.b,G[2]*v.a+G[3]*v.b};
  if(report){double q=v.a*sin(theta)+v.b*cos(theta);if(q>0){double dq=dv.a*sin(theta)+dv.b*cos(theta);fd1+=dv.c*q+v.c*dq;fd2+=dv.d*q+v.d*dq;}maxbalance=max(maxbalance,abs(v.a*v.a+v.b*v.b-v.c*v.c-v.d*v.d));}
  double *aa=(double*)&acc[i],*vv=(double*)&state[i],*dd=(double*)&dst[i],*gg=(double*)&dv;
  if(mode==0){for(int j=0;j<4;j++){aa[j]=vv[j]+acccoef*gg[j];dd[j]=vv[j]+basecoef*gg[j];}}
  else if(mode==1){for(int j=0;j<4;j++){aa[j]+=acccoef*gg[j];dd[j]=vv[j]+basecoef*gg[j];}}
  else if(mode==2){for(int j=0;j<4;j++)dd[j]=aa[j]+basecoef*gg[j];}
 }
 if(report){
  double u=sqrt(uu),E=(pow(f1-sin(theta),2)+pow(f2-cos(theta),2))/2,dE=(f1-sin(theta))*fd1+(f2-cos(theta))*fd2;
  double direction=sqrt(max(0.,2-2*uv/u));
  simlog<<t<<','<<At*At<<','<<E<<','<<dE<<','<<coremass<<','<<atan(tanU)<<','<<atan(tanW)<<','<<S<<','<<weak2<<','<<f1<<','<<f2<<','<<mm<<','<<abs(RR)/u<<','<<sqrt(pp)/u<<','<<abs(log(u/At))<<','<<direction<<','<<sqrt(cdiff)<<','<<sqrt(bdiff)<<','<<sqrt(ddiff)<<','<<loss<<','<<maxbalance<<','<<uncertain[0].size()<<','<<uncertain[1].size()<<','<<maxJD<<','<<maxJB<<','<<maxJP<<','<<coregamma<<','<<maxv<<','<<mincorea<<'\n';simlog.flush();
 }
}
int main(int argc,char**argv){
 double dtmax=argc>1?stod(argv[1]):.02;string tag=argc>2?argv[2]:"coarse";
 auto wall0=chrono::steady_clock::now();mt19937_64 rng(20260918);uniform_real_distribution<double> uni(0,2*acos(-1.));normal_distribution<double> normal(0,1);
 rmin[0]=rmin[1]=1e100;rmax[0]=rmax[1]=-1e100;double maxnoise=0;
 for(int j=0;j<N;j++){int g=j>=N/2;double nx=1e-9*normal(rng),ny=1e-9*normal(rng);training[j]={nx+(g?0:1),ny+(g?1e-4:0),0,0};double r=g?training[j].x/training[j].y:training[j].y/training[j].x;rmin[g]=min(rmin[g],r);rmax[g]=max(rmax[g],r);maxnoise=max(maxnoise,hypot(nx,ny));}
 double Q=0,T=0,C=0,K=0,coreQ=0,R=0;size_t corecount=0;
 array<double,26> initnet{};
 for(size_t i=0;i<h;i++){
  double angle=uni(rng),a=ep*cos(angle),b=ep*sin(angle);state[i]={a,b,a,b};initial[i]={a,b};
  Q+=pow(max(0.,a),2);T+=max(0.,a)*max(0.,b);C+=a*max(0.,b);K+=pow(max(0.,b),2);R+=b*max(0.,a);
  if(a>=abs(b)){corecount++;coreQ+=a*a;}
  for(int j=0;j<13;j++){double x,y;if(j==11){x=1;y=0;}else if(j==12){x=0;y=1;}else{double th=3.5e-5+j*1e-6;x=sin(th);y=cos(th);}double q=max(0.,a*x+b*y);initnet[2*j]+=a*q;initnet[2*j+1]+=b*q;}
 }
 A0=sqrt(Q);
 if(h<1000){
  for(size_t i=0;i<min(h,(size_t)40);i++){double ang=(i%2?acos(-1.)/2:0)+(double(i)-20)*(i%2?1e-10:1e-6);state[i]={ep*cos(ang),ep*sin(ang),ep*cos(ang),ep*sin(ang)};}
  stage(state,tmp,1,0,0,0,false);double output_error=0,gradient_error=0;
  for(auto &x:training){double y1=0,y2=0;for(const auto &n:state){double q=max(0.,n.a*x.x+n.b*x.y);y1+=n.c*q;y2+=n.d*q;}output_error=max(output_error,max(abs((x.x-y1)-x.rx),abs((x.y-y2)-x.ry)));}
  for(size_t i=0;i<h;i++){const auto &v=state[i];Neuron g{};for(const auto &x:training){double q=v.a*x.x+v.b*x.y;if(q>0){g.c+=x.rx*q/N;g.d+=x.ry*q/N;double z=(x.rx*v.c+x.ry*v.d)/N;g.a+=z*x.x;g.b+=z*x.y;}}gradient_error=max(gradient_error,max({abs(tmp[i].a-v.a-g.a),abs(tmp[i].b-v.b-g.b),abs(tmp[i].c-v.c-g.c),abs(tmp[i].d-v.d-g.d)}));}
  cout<<setprecision(17)<<"preflight output error "<<output_error<<" gradient error "<<gradient_error<<" boundary neurons "<<uncertain[0].size()<<","<<uncertain[1].size()<<"\n";return (output_error<1e-13&&gradient_error<1e-13)?0:3;
 }
 double maxnet=0;for(int j=0;j<13;j++){double x,y;if(j==11){x=1;y=0;}else if(j==12){x=0;y=1;}else{double th=3.5e-5+j*1e-6;x=sin(th);y=cos(th);}maxnet=max(maxnet,hypot(initnet[2*j]-.01*x,initnet[2*j+1]-.01*y));}
 ofstream manifest("outputs/SWINGBY_SIM_"+tag+"_initial.txt");manifest<<setprecision(17)<<"h "<<h<<"\nepsilon "<<ep<<"\nmaxnoise "<<maxnoise<<"\nQ/h "<<Q/.04<<"\nT/h "<<T/.04<<"\nC/h "<<C/.04<<"\nK/h "<<K/.04<<"\ncorecount/h "<<(double)corecount/h<<"\ncoreQ/h "<<coreQ/.04<<"\nb0 "<<abs(R)/A0<<"\nmax_net_error "<<maxnet<<'\n';
 if(!(maxnoise<1e-8&&Q/.04>.248&&Q/.04<.252&&abs(C/.04)<.001&&K/.04>.248&&K/.04<.252&&abs(T/.04-1/(4*acos(-1.)))<.002&&corecount>.24*h&&coreQ/.04>1./8+1/(4*acos(-1.))-.002&&abs(R)/A0<1.25*ep&&maxnet<2.72721e-5)){cerr<<"Initial certificate event failed; not retrying seed.\n";return 2;}
 vector<double> targets{.00017};for(double q:{.69,.71,.715,.95})targets.push_back(log(q/(1-q)*(1-Q)/Q));double end=targets.back();manifest<<"terminal "<<end<<"\ndtmax "<<dtmax<<'\n';manifest.close();
 simlog.open("outputs/SWINGBY_SIM_"+tag+".csv");simlog<<setprecision(17)<<"t,A2,E,dE,core_mass,angle_u,angle_w,S,weak_fit,f1,f2,m,z,l,log_u_A,direction_error,c_error,B_movement,D_movement,loss,balance_error,strong_boundary,weak_boundary,J_D_observed,J_B_observed,J_p_observed,Gamma_core_observed,norm_factor_observed,min_core_a\n";
 cerr<<"Initial event passed; beginning exact empirical gradients; terminal "<<end<<".\n";
 double t=0;int step=0;
 while(t<end-1e-12){double dt=min(dtmax,end-t);if(t<.00017-1e-12)dt=min(dt,1e-5);else if(t<.1)dt=min(dt,.001);for(double target:targets)if(target>t+1e-12)dt=min(dt,target-t);
  stage(state,tmp,dt/2,dt/6,0,t,true);stage(tmp,tmp,dt/2,dt/3,1,t+dt/2,false);stage(tmp,tmp,dt,dt/3,1,t+dt/2,false);stage(tmp,state,dt/6,0,2,t+dt,false);t+=dt;step++;
  if(step%25==0)cerr<<"step "<<step<<" time "<<t<<" elapsed "<<chrono::duration<double>(chrono::steady_clock::now()-wall0).count()<<"s\n";
 }
 stage(state,tmp,0,0,3,t,true);cerr<<"Finished "<<step<<" steps in "<<chrono::duration<double>(chrono::steady_clock::now()-wall0).count()<<" seconds.\n";
}
