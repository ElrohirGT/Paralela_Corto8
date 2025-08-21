#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

static double slope_last_k(const double *a,int T,int K){
    if(K<2) return 0.0;
    if(K>T) K=T;
    int start=T-K;
    double sumx=0,sumy=0,sumxy=0,sumx2=0;
    for(int i=0;i<K;i++){double x=i,y=a[start+i];sumx+=x;sumy+=y;sumxy+=x*y;sumx2+=x*x;}
    double denom=K*sumx2 - sumx*sumx;
    if(fabs(denom)<1e-12) return 0.0;
    return (K*sumxy - sumx*sumy)/denom;
}
static void push_next(double *a,int T,double next){
    for(int i=0;i<T-1;i++) a[i]=a[i+1];
    a[T-1]=next;
}

int main(int argc,char**argv){
    if(argc<2){fprintf(stderr,"Uso: %s F < datos.txt\n",argv[0]);return 1;}
    int F=atoi(argv[1]);

    // Leer datos
    int cap=1024,n=0,maxRegion=-1;
    int *regions=(int*)malloc(cap*sizeof(int));
    double *all=(double*)malloc(cap*3*sizeof(double));

    while(1){
        int reg; double t,h,v;
        if(scanf("%d %lf %lf %lf",&reg,&t,&h,&v)!=4) break;
        if(n>=cap){cap*=2;regions=realloc(regions,cap*sizeof(int));all=realloc(all,cap*3*sizeof(double));}
        regions[n]=reg;
        all[3*n]=t; all[3*n+1]=h; all[3*n+2]=v;
        if(reg>maxRegion) maxRegion=reg;
        n++;
    }
    if(n==0){fprintf(stderr,"Archivo vacío\n");return 2;}

    int R=maxRegion+1;
    int T=n/R;
    int K=T;

    double **temp=(double**)malloc(R*sizeof(*temp));
    double **hum =(double**)malloc(R*sizeof(*hum));
    double **wind=(double**)malloc(R*sizeof(*wind));
    for(int r=0;r<R;r++){temp[r]=malloc(T*sizeof(double));hum[r]=malloc(T*sizeof(double));wind[r]=malloc(T*sizeof(double));}

    int *count=(int*)calloc(R,sizeof(int));
    for(int i=0;i<n;i++){
        int r=regions[i];
        int pos=count[r]++;
        temp[r][pos]=all[3*i];
        hum[r][pos]=all[3*i+1];
        wind[r][pos]=all[3*i+2];
    }
    free(all); free(regions); free(count);

    double t0=omp_get_wtime();

    for(int f=0;f<F;f++){
        #pragma omp parallel for
        for(int r=0;r<R;r++){
            double mT=slope_last_k(temp[r],T,K);
            double mH=slope_last_k(hum[r], T,K);
            double mV=slope_last_k(wind[r],T,K);
            double t_pred=temp[r][T-1]+mT;
            double h_pred=hum[r][T-1]+mH;
            double v_pred=wind[r][T-1]+mV;
            if(h_pred<0)h_pred=0; if(h_pred>100)h_pred=100;
            push_next(temp[r],T,t_pred);
            push_next(hum[r], T,h_pred);
            push_next(wind[r],T,v_pred);
            #pragma omp critical
            printf("Region %d -> Dia+%d: Temp=%.2f Hum=%.2f Viento=%.2f\n",
                   r,f+1,t_pred,h_pred,v_pred);
        }
    }

    double t1=omp_get_wtime();
    printf("Tiempo total (paralelo,%d hilos)=%.6f s\n",omp_get_max_threads(),t1-t0);

    for(int r=0;r<R;r++){free(temp[r]);free(hum[r]);free(wind[r]);}
    free(temp);free(hum);free(wind);
    return 0;
}
