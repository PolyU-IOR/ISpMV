#pragma once
#include "matrix_io.hpp"
#include "numerical_vectors.hpp"
#include "numerical_checks.hpp"
#include <ispmv/ispmv.h>
#define CUSPARSE_ENABLE_EXPERIMENTAL_API
#include <cusparse.h>
#include <functional>
#include <iostream>
#include <string>

static void vendor_check(cusparseStatus_t status) {
    if (status != CUSPARSE_STATUS_SUCCESS)
        throw std::runtime_error(std::string("cuSPARSE: ") + cusparseGetErrorString(status));
}
static void product_check(ispmv_status status) {
    if (status != ISPMV_SUCCESS) throw std::runtime_error(ispmv_error_message());
}
static double elapsed_ms(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
struct Events {
    cudaEvent_t begin=nullptr,end=nullptr;
    Events(){CUDA_CHECK(cudaEventCreate(&begin));CUDA_CHECK(cudaEventCreate(&end));}
    ~Events(){cudaEventDestroy(begin);cudaEventDestroy(end);}
    double measure(const std::function<void()>& call,cudaStream_t stream,int count) {
        CUDA_CHECK(cudaEventRecord(begin,stream));
        for(int i=0;i<count;++i)call();
        CUDA_CHECK(cudaEventRecord(end,stream));CUDA_CHECK(cudaEventSynchronize(end));
        float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,begin,end));return ms*1000.0/count;
    }
};
struct Operator {
    cudaStream_t stream=nullptr;
    int *offsets=nullptr,*columns=nullptr;
    double *values=nullptr,*x=nullptr,*y=nullptr;
    ispmv_plan* native=nullptr;
    ispmv_device* hardware=nullptr;
    bool owns_hardware=false;
    double device_initialization_ms=0;
    cusparseHandle_t handle=nullptr;
    cusparseSpMatDescr_t matrix=nullptr;
    cusparseDnVecDescr_t dx=nullptr,dy=nullptr,dz=nullptr;
    cusparseSpMVOpDescr_t descriptor=nullptr;
    cusparseSpMVOpPlan_t operation=nullptr;
    void* workspace=nullptr;size_t workspace_bytes=0;
    std::string method;
    ~Operator(){
        if(stream)cudaStreamSynchronize(stream);
        if(native)ispmv_release(native);
        if(owns_hardware)ispmv_device_release(hardware);
        if(operation)cusparseSpMVOp_destroyPlan(operation);
        if(descriptor)cusparseSpMVOp_destroyDescr(descriptor);
        if(dx)cusparseDestroyDnVec(dx);if(dy)cusparseDestroyDnVec(dy);if(dz)cusparseDestroyDnVec(dz);
        if(matrix)cusparseDestroySpMat(matrix);if(handle)cusparseDestroy(handle);
        cudaFree(workspace);cudaFree(y);cudaFree(x);cudaFree(values);cudaFree(columns);cudaFree(offsets);
        if(stream)cudaStreamDestroy(stream);
    }
    void upload(const HostCsr& a, ispmv_device* shared_hardware=nullptr) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&offsets),a.row_offsets.size()*sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&columns),a.column_indices.size()*sizeof(int)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&values),a.values.size()*sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&x),size_t(a.cols)*sizeof(double)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&y),size_t(a.rows)*sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(offsets,a.row_offsets.data(),a.row_offsets.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
        CUDA_CHECK(cudaMemcpyAsync(columns,a.column_indices.data(),a.column_indices.size()*sizeof(int),cudaMemcpyHostToDevice,stream));
        CUDA_CHECK(cudaMemcpyAsync(values,a.values.data(),a.values.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
        std::vector<double> initial(a.cols);vector_input(initial,0);
        CUDA_CHECK(cudaMemcpyAsync(x,initial.data(),initial.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
        CUDA_CHECK(cudaMemsetAsync(y,0,size_t(a.rows)*sizeof(double),stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        // Library handles are analogous to the already-existing solver context.
        if(method=="ispmv"){
            hardware=shared_hardware;
            if(!hardware){auto begin=std::chrono::steady_clock::now();product_check(ispmv_device_create(-1,&hardware));
                owns_hardware=true;device_initialization_ms=elapsed_ms(begin);}
        }
        if(method!="ispmv"){vendor_check(cusparseCreate(&handle));vendor_check(cusparseSetStream(handle,stream));}
    }
    double prepare(const HostCsr& a) {
        const auto start=std::chrono::steady_clock::now();
        if(method=="ispmv") {
            ispmv_csr csr;ispmv_csr_init(&csr);csr.rows=a.rows;csr.columns=a.cols;csr.nonzeros=a.nnz;
            csr.device_row_offsets=offsets;csr.device_column_indices=columns;csr.device_values=values;
            ispmv_config config;ispmv_config_init(&config);config.copy_device_matrix=0;
            product_check(ispmv_prepare(hardware,&csr,&config,&native));
        } else {
            vendor_check(cusparseCreateCsr(&matrix,a.rows,a.cols,a.nnz,offsets,columns,values,
                CUSPARSE_INDEX_32I,CUSPARSE_INDEX_32I,CUSPARSE_INDEX_BASE_ZERO,CUDA_R_64F));
            vendor_check(cusparseCreateDnVec(&dx,a.cols,x,CUDA_R_64F));
            vendor_check(cusparseCreateDnVec(&dy,a.rows,y,CUDA_R_64F));
            if(method=="cusparse_alg2") {
                const double alpha=1,beta=0;
                vendor_check(cusparseSpMV_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,matrix,
                    dx,&beta,dy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG2,&workspace_bytes));
                if(workspace_bytes)CUDA_CHECK(cudaMalloc(&workspace,workspace_bytes));
                vendor_check(cusparseSpMV_preprocess(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,matrix,
                    dx,&beta,dy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG2,workspace));
            } else {
                vendor_check(cusparseCreateDnVec(&dz,a.rows,y,CUDA_R_64F));
                const auto algorithm=method=="spmvop_alg1"?CUSPARSE_SPMVOP_ALG1:CUSPARSE_SPMVOP_ALG2;
                vendor_check(cusparseSpMVOp_bufferSize(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,matrix,
                    dx,dy,dz,CUDA_R_64F,algorithm,&workspace_bytes));
                if(workspace_bytes)CUDA_CHECK(cudaMalloc(&workspace,workspace_bytes));
                vendor_check(cusparseSpMVOp_createDescr(handle,&descriptor,CUSPARSE_OPERATION_NON_TRANSPOSE,
                    matrix,dx,dy,dz,CUDA_R_64F,algorithm,workspace));
                vendor_check(cusparseSpMVOp_createPlan(handle,descriptor,&operation,nullptr,0));
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());return elapsed_ms(start);
    }
    void apply(double alpha,double beta) {
        if(native)product_check(ispmv_apply(native,alpha,x,beta,y,stream));
        else if(operation)vendor_check(cusparseSpMVOp(handle,operation,&alpha,&beta,dx,dy,dz));
        else vendor_check(cusparseSpMV(handle,CUSPARSE_OPERATION_NON_TRANSPOSE,&alpha,matrix,dx,
            &beta,dy,CUDA_R_64F,CUSPARSE_SPMV_CSR_ALG2,workspace));
    }
};

inline Json measure_operator(Operator& op,const HostCsr& a,Events& events,double prepare_ms,
    const char* label,const char* input,const char* preparation) {
    const double scalars[2][2]={{1,0},{1.25,-.375}};
    Json::Array checks[2],samples[2];bool accepted[2]={true,true};double first_us[2]{},first_wall_us[2]{};
    std::function<void()> call[2];
    for(int m=0;m<2;++m){call[m]=[&,m]{op.apply(scalars[m][0],scalars[m][1]);};
        CUDA_CHECK(cudaMemsetAsync(op.y,0,size_t(a.rows)*sizeof(double),op.stream));CUDA_CHECK(cudaStreamSynchronize(op.stream));
        const auto start=std::chrono::steady_clock::now();first_us[m]=events.measure(call[m],op.stream,1);first_wall_us[m]=1000*elapsed_ms(start);}
    std::vector<double> x(a.cols),old(a.rows),y(a.rows),again(a.rows);
    std::vector<long double> dot(a.rows),norm(a.rows),ref(a.rows),scale(a.rows);
    for(int trial=0;trial<9;++trial){
        vector_input(x,trial);
        #pragma omp parallel for schedule(dynamic,256)
        for(int r=0;r<a.rows;++r){long double sum=0,mag=0;
            for(int j=a.row_offsets[r];j<a.row_offsets[r+1];++j){const long double term=static_cast<long double>(a.values[j])*x[a.column_indices[j]];sum+=term;mag+=std::abs(term);}
            dot[r]=sum;norm[r]=mag;}
        CUDA_CHECK(cudaMemcpyAsync(op.x,x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice,op.stream));
        for(int m=0;m<2;++m){const double alpha=scalars[m][0],beta=scalars[m][1];
            for(int r=0;r<a.rows;++r){old[r]=beta==0?std::numeric_limits<double>::quiet_NaN():std::sin(.017*(r+1));
                const long double previous=beta==0?0:static_cast<long double>(beta)*old[r];
                ref[r]=alpha*dot[r]+previous;scale[r]=std::abs(alpha)*norm[r]+std::abs(previous);}
            for(auto* target:{&y,&again}){CUDA_CHECK(cudaMemcpyAsync(op.y,old.data(),old.size()*sizeof(double),cudaMemcpyHostToDevice,op.stream));call[m]();
                CUDA_CHECK(cudaMemcpyAsync(target->data(),op.y,target->size()*sizeof(double),cudaMemcpyDeviceToHost,op.stream));CUDA_CHECK(cudaStreamSynchronize(op.stream));}
            auto check=check_result(y,again,ref,scale,trial);
            accepted[m]&=std::get<bool>(std::get<Json::Object>(check.value).at("accepted").value);checks[m].push_back(std::move(check));
        }
    }
    vector_input(x,0);CUDA_CHECK(cudaMemcpyAsync(op.x,x.data(),x.size()*sizeof(double),cudaMemcpyHostToDevice,op.stream));
    Json::Array modes;
    for(int m=0;m<2;++m){CUDA_CHECK(cudaMemsetAsync(op.y,0,size_t(a.rows)*sizeof(double),op.stream));
        for(int i=0;i<100;++i)call[m]();CUDA_CHECK(cudaStreamSynchronize(op.stream));
        std::vector<double> times;
        for(int group=0;group<3;++group){for(int i=0;i<10;++i)call[m]();CUDA_CHECK(cudaStreamSynchronize(op.stream));
            const double us=events.measure(call[m],op.stream,50);times.push_back(us);samples[m].push_back(us);}
        CUDA_CHECK(cudaMemcpy(y.data(),op.y,y.size()*sizeof(double),cudaMemcpyDeviceToHost));
        const bool finite=std::all_of(y.begin(),y.end(),[](double v){return std::isfinite(v);});
        std::sort(times.begin(),times.end());
        modes.push_back(Json::Object{{"alpha",scalars[m][0]},{"beta",scalars[m][1]},
            {"accepted",accepted[m]&&finite},{"validation",checks[m]},{"samples_us",samples[m]},
            {"steady_spmv_event_us",times[1]},{"first_after_scalar_change_event_us",first_us[m]},
            {"first_after_scalar_change_wall_us",first_wall_us[m]},{"timed_recurrence_finite",finite}});
    }
    int device=0,runtime=0,driver=0,vendor=0;cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDevice(&device));CUDA_CHECK(cudaGetDeviceProperties(&props,device));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));CUDA_CHECK(cudaDriverGetVersion(&driver));
    if(op.handle)vendor_check(cusparseGetVersion(op.handle,&vendor));
    Json result=Json::Object{{"schema",2},{"label",label},{"input",input},{"method",op.method},
        {"rows",a.rows},{"columns",a.cols},{"nnz",a.nnz},{"gpu",props.name},
        {"cuda_runtime",runtime},{"cuda_driver",driver},{"cusparse_version",vendor},
        {"plan_build_wall_ms",prepare_ms},{"device_csr_ready",true},{"preparation",preparation},
        {"warmup",100},{"groups",3},{"calls_per_group",50},{"group_prewarm",10},
        {"vendor_workspace_bytes",int64_t(op.workspace_bytes)},{"modes",modes}};
    return result;
}
