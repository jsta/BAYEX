import pystan

def max_stable_model():

    modelCode = """    
    
    functions{
        
        real gev_lpdf(vector y, vector mu, vector sigma, real xi){
            int n = rows(y);
            vector[n] log_z = log1p(xi * (y - mu) ./ sigma) * (-1/xi);
            return -sum(log(sigma)) + (1 + xi) * sum(log_z) - sum(exp(log_z));
        }
        
        real gev_xi_lpdf(vector y, vector mu, vector sigma, vector xi){
            int n = rows(y);
            vector[n] log_z = -log1p(xi .* (y - mu) ./ sigma) ./ xi;
            return -sum(log(sigma)) + sum((1 + xi) .* log_z) - sum(exp(log_z));
        }
        
        matrix kernels(matrix r, real tau,row_vector row_ones){
            int nx = cols(r);
            int nl = rows(r);
            matrix[nl,nx] wi;
            row_vector[nx] wn; // normalization
            wi = exp(-0.6931/square(tau)*r);
            wn = row_ones * wi;
            return wi ./ rep_matrix(wn,nl);
        }
        
        real PS_lpdf(matrix AB, real alpha,int nr,int nc){
            matrix[nr,nc] piB;
            matrix[nr,nc] logc;
            matrix[nr,nc] logA;
            piB = pi()*AB[:,(nc+1):2*nc];
            logc = 1/(1-alpha)*log(sin(alpha*piB) ./ sin(piB)) + log(sin((1-alpha)*piB) ./ sin(alpha*piB));
            logA = log(AB[:,1:nc]);
            return nr*nc*log(alpha/(1-alpha)) - 1/(1-alpha)*sum(logA) + sum(logc) - sum(exp(logc -alpha/(1-alpha)*logA));
        }
        
        real logPS_lpdf(matrix AB, real alpha,int nr,int nc,real scale_PS){
            matrix[nr,nc] piB;
            matrix[nr,nc] logc;
            matrix[nr,nc] A;
            real b = scale_PS * alpha/(1-alpha);
            piB = pi()*AB[:,(nc+1):2*nc];
            logc = 1/(1-alpha)*log(sin(alpha*piB) ./ sin(piB)) + log(sin((1-alpha)*piB) ./ sin(alpha*piB));
            A = (-b)*AB[:,1:nc];
            return nr*nc*log(b) + sum(A) + sum(logc) - sum(exp(logc + A));
        }
        
        
        real euclidean_dist(real x1, real y1, real x2, real y2){
            real d;
            if(x1 == x2 && y1 == y2)
                d = 0;
            else
                d = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1))/10;
            return d;
        }
        
        real chordal_dist(real x1, real y1, real x2, real y2){
            real d2r = pi()/180;
            real geo_dist;
            if(x1 == x2 && y1 == y2)
                geo_dist = 0;
            else
                geo_dist = 6371*acos(sin(d2r*y1) * sin(d2r*y2) + cos(d2r*y1) * cos(d2r*y2) * cos(d2r*(x2 - x1))); //geodesic distance
            return 2*6371*sin(geo_dist/(2*6371))/1000; // chordal distance
        }
        
        matrix matern_52_mat(matrix r,matrix r_sq, real sig, real l){
            return square(sig)*(1 + (sqrt(5)/l)*r + (5/(3*square(l)))*r_sq) .* exp((-sqrt(5)/l)*r);
        }
        
        vector matern_52_vec(vector r,vector r_sq, real sig, real l){
            return square(sig)*(1 + (sqrt(5)/l)*r + (5/(3*square(l)))*r_sq) .* exp((-sqrt(5)/l)*r);
        } 
        
        real matern_52_scalar(real r,real r_sq, real sig, real l){
            return square(sig)*(1 + (sqrt(5)/l)*r + (5/(3*square(l)))*r_sq) * exp((-sqrt(5)/l)*r);
        }   
                
        vector nngp_chol_vec(matrix dx1,matrix dx2,matrix dx1_sq,matrix dx2_sq,real s,real l,int[,] indM,int M,int n,
                             vector a,real delta){
            matrix[M*M,n-1] C1;
            matrix[M,n-1] C2;
            vector[n] D;
            vector[n] ac;
            real ssq = square(s);
            real Aii;
            real delta_sqrt = sqrt(delta);
            matrix[M,M] Ddelta = diag_matrix(rep_vector(delta,M));
            
            C1[1,1] = matern_52_scalar(dx1[1,1],dx1_sq[1,1],s,l);
            C2[1,1] = matern_52_scalar(dx2[1,1],dx2_sq[1,1],s,l);
            for(i in 2:(M-1)){
                C1[1:(i*i),i] = matern_52_vec(dx1[1:(i*i),i],dx1_sq[1:(i*i),i],s,l);
                C2[1:i,i] = matern_52_vec(dx2[1:i,i],dx2_sq[1:i,i],s,l);
            }
            C1[:,M:] = matern_52_mat(dx1[:,M:],dx1_sq[:,M:],s,l);
            C2[:,M:] = matern_52_mat(dx2[:,M:],dx2_sq[:,M:],s,l);
            
            ac[1] = a[1]*(s + delta_sqrt);
            D[1] = 1/(s + delta_sqrt);
            Aii = -C2[1,1]/(C1[1,1] + delta);
            D[2] = 1/sqrt(ssq + C2[1,1] * Aii);
            ac[2] = a[2]/D[2] - Aii*ac[1];
            for(i in 3:n){
                int ie = (i-1)<M ? (i-1) : M;
                row_vector[ie] Ai;
                matrix[ie,ie] Li;
                Li = cholesky_decompose(to_matrix(C1[1:ie*ie,i-1],ie,ie) + Ddelta[1:ie,1:ie]);
                Ai = -mdivide_right_tri_low(mdivide_left_tri_low(Li, C2[1:ie,i-1])',Li);
                D[i] = 1/sqrt(ssq + delta + Ai*C2[1:ie,i-1]);
                ac[i] = a[i]/D[i] - Ai*ac[indM[i-1,1:ie]];
            }
            return ac;
        }
        
        matrix nngp_chol_mat(matrix dx1, matrix dx2,matrix dx1_sq,matrix dx2_sq, real s, real l, int[,] indM, int M, int n,
                             matrix a,real delta){
            matrix[M*M,n-1] C1;
            matrix[M,n-1] C2;
            matrix[rows(a),n] ac;
            vector[n] D;
            int no = cols(a);
            real ssq = square(s);
            real Aii;
            real delta_sqrt = sqrt(delta);
            matrix[M,M] Ddelta = diag_matrix(rep_vector(delta,M));
            
            C1[1,1] = matern_52_scalar(dx1[1,1],dx1_sq[1,1],s,l);
            C2[1,1] = matern_52_scalar(dx2[1,1],dx2_sq[1,1],s,l);
            for(i in 2:(M-1)){
                C1[1:(i*i),i] = matern_52_vec(dx1[1:(i*i),i],dx1_sq[1:(i*i),i],s,l);
                C2[1:i,i] = matern_52_vec(dx2[1:i,i],dx2_sq[1:i,i],s,l);
            }
            C1[:,M:] = matern_52_mat(dx1[:,M:],dx1_sq[:,M:],s,l);
            C2[:,M:] = matern_52_mat(dx2[:,M:],dx2_sq[:,M:],s,l);
            
            ac[:,1] = a[:,1]*(s + delta_sqrt); 
            D[1] = 1/(s + delta_sqrt);
            Aii = -C2[1,1]/(C1[1,1] + delta);
            D[2] = 1/sqrt(ssq + C2[1,1] * Aii);
            ac[:,2] = a[:,2]/D[2] - Aii*ac[:,1];
            for(i in 3:n){
                int ie = (i-1)<M ? (i-1) : M;
                row_vector[ie] Ai;
                matrix[ie,ie] Li;
                Li = cholesky_decompose(to_matrix(C1[1:ie*ie,i-1],ie,ie) + Ddelta[1:ie,1:ie]);
                Ai = -mdivide_right_tri_low(mdivide_left_tri_low(Li, C2[1:ie,i-1])',Li);
                D[i] = 1/sqrt(ssq + delta + Ai*C2[1:ie,i-1]);
                ac[:,i] = a[:,i]/D[i] - ac[:,indM[i-1,1:ie]]*Ai';   
            }
           
            return ac';
        }
        
    }
    
    data {
        int<lower=1> no; // number of years
        int<lower=1> nx; // number of data sites
        int<lower=1> nl; // number of knots
        int<lower=1> nc; // number of covariates
        int<lower=1> Nok;
        int<lower=0,upper=1> nngp;
        int<lower=0> Mgp; //number of nearest neighbors for GPs
        int<lower=0> Mknot; // number of nearest neighbor residual spatial process
        int<lower=0,upper=1> nnknot;
        int<lower=1> grainsize;
        int<lower=0,upper=1> is_gumbel;
        int<lower=0,upper=1> trend_linear;
        int<lower=0,upper=1> xi_spatial;
        vector[Nok] y_ok; // valid data
        int kok[Nok]; // indices of valid data
        matrix[nx,nc] xp; // covariates at data sites
        vector[nx] lon_data;
        vector[nx] lat_data;
        vector[nl] lon_knot;
        vector[nl] lat_knot;
    }
    
    transformed data{
        int nxi = is_gumbel == 0? 1 : 0;
        int nxi2 = xi_spatial == 1? nx : 1;
        int ntrend = trend_linear == 1? 1 : no;
        int nb0 = trend_linear == 1? 0 : no;
        int nbs = trend_linear == 1? 0 : 1;
        int M = (nngp == 0) ? 0 : Mgp;
        int Mk = (nnknot == 0) ? 0 : Mknot;
        int nli = (nnknot == 0) ? nl : Mk;
        real scale_b00 = trend_linear == 1? 100 : 500;
        matrix[nli,nx] dl;
        matrix[(nngp == 0) ? nx : 0,nx] dx;
        matrix[(nngp == 0) ? nx : 0,nx] dx_sq;
        matrix[M*M,nx-1] dx1 = rep_matrix(0,M*M,nx-1);
        matrix[M,nx-1] dx2 = rep_matrix(0,M,nx-1);
        matrix[M*M,nx-1] dx1_sq;
        matrix[M,nx-1] dx2_sq;
        int indM[nx-1,M];
        int indt[nx,no];
        int indMk[nx,Mk];
        real dtmp;
        row_vector[no] t;
        real delta = 1e-12; // used to deal with non-positive definiteness
        row_vector[nli] row_ones = rep_row_vector(1,nli); // used in kernels
        real scale_PS = 2;
        
        for(i in 1:nx){
            for(j in 1:no){
                indt[i,j] = i + (j-1)*nx;
            }
        }
        
        for(i in 1:no)
            t[i] = (i - 1)/1.;
        
        if(nngp == 0){
            for(i in 1:nx){
                for(j in 1:nx){
                    if(j>i)
                        dx[i,j] = euclidean_dist(lon_data[i],lat_data[i],lon_data[j],lat_data[j]);
                    else if(j == i)
                        dx[i,j] = 0.;
                    else{
                        dtmp = dx[j,i];
                        dx[i,j] = dtmp;
                    }
                }
            }
            dx_sq = dx .* dx;
        } 
        
        if(nngp == 1){
            for(i in 1:nx-1){
                if(i<=M){
                    for(j in 1:i){
                        indM[i,j] = j; 
                    }
                }
                else{
                    vector[i] dxi;
                    for(j in 1:i)
                        dxi[j] = euclidean_dist(lon_data[j],lat_data[j],lon_data[i+1],lat_data[i+1]);
                    indM[i,:] = sort_asc(sort_indices_asc(dxi)[1:M]);
                }
            }
        
            for(i in 1:nx-1){
                if(i==1){
                    dx1[1,i] = 0;
                    dx2[1,i] = euclidean_dist(lon_data[1],lat_data[1],lon_data[2],lat_data[2]);
                }
                else{
                    int ie = i<M ? i : M;
                    int ct = 0;
                    for(j in 1:ie){
                        for(k in 1:ie){
                            ct += 1;
                            dx1[ct,i] = euclidean_dist(lon_data[indM[i,j]],lat_data[indM[i,j]],lon_data[indM[i,k]],lat_data[indM[i,k]]);
                        }
                        dx2[j,i] = euclidean_dist(lon_data[indM[i,j]],lat_data[indM[i,j]],lon_data[i+1],lat_data[i+1]);
                    }
                }   
            }
            dx1_sq = dx1 .* dx1;
            dx2_sq = dx2 .* dx2;
        }
        
        if(nnknot == 0){
            for(i in 1:nl){
                for(j in 1:nx){
                    dl[i,j] = euclidean_dist(lon_knot[i],lat_knot[i],lon_data[j],lat_data[j]);
                    dtmp = dl[i,j];
                    dl[i,j] = dtmp * dtmp;
                }
            }
        }
        else{
            for(i in 1:nx){
                vector[nl] dxi;
                for(j in 1:nl){
                    dxi[j] = euclidean_dist(lon_knot[j],lat_knot[j],lon_data[i],lat_data[i]);
                }
                indMk[i,:] = sort_asc(sort_indices_asc(dxi)[1:Mk]);
            }
        
            for(i in 1:nx){
                for(j in 1:Mk){
                    dl[j,i] = euclidean_dist(lon_knot[indMk[i,j]],lat_knot[indMk[i,j]],lon_data[i],lat_data[i]);
                    dtmp = dl[j,i];
                    dl[j,i] = dtmp * dtmp;
                }
            }
        }
        
    }
    
    parameters {
        vector[nx] a_b10;
        vector[nx*nb0] a_b0;
        vector[nx] a_sigma;
        vector[nx] a_b00;
        vector[nxi * nxi2] a_xi;
        //matrix<lower=0>[no,nl] A;
        matrix[no,nl] logA;
        matrix<lower=0,upper=1>[no,nl] B;
        vector[nc] beta_b1;
        vector[nc] beta_s;
        real<lower=0,upper=5> alpha_raw;
        real<lower=0> s_b10;
        real<lower=0> l_b10;
        real<lower=0> s_b0[nbs];
        real<lower=0> l_b0[nbs];
        real<lower=0> s_b00;
        real<lower=0> l_b00;
        real<lower=0> l_xi[xi_spatial*nxi];
        real<lower=0> s_sigma;
        real<lower=0> l_sigma;
        real<lower=0> s_xi[nxi];
        real<lower=0> tau;
    }
    transformed parameters {
        vector[nx*no] b1;
        vector[nx] b10;
        vector[nx] b00;
        matrix[nx,nb0] b0;
        matrix[nx,nb0] bi0;
        vector[nx] sigma;
        vector[nx*no] mu_star;
        vector[nx*no] sigma_star;
        vector[nxi*nxi2] xi;
        vector[nxi*nxi2] xi_star;
        matrix[no,nl] A = exp(logA*scale_PS);
        real alpha = alpha_raw/5;
        
        if(nngp == 0){
            matrix[nx,nx] L;
            matrix[nx,nx] Q;
            
            Q = matern_52_mat(dx,dx_sq,s_b10,l_b10);
            for(i in 1:nx)
                Q[i,i] += delta; // deal with non-positive definiteness
            L = cholesky_decompose(Q);
            b10 = xp * beta_b1 + L * a_b10;
            
            Q = matern_52_mat(dx,dx_sq,s_b00/scale_b00,l_b00);
            for(i in 1:nx)
                Q[i,i] += delta;
            L = cholesky_decompose(Q);
            b00 = L * a_b00;
                        
            Q = matern_52_mat(dx,dx_sq,s_sigma,l_sigma);
            for(i in 1:nx)
                Q[i,i] += delta;
            L = cholesky_decompose(Q);
            sigma = exp(xp * beta_s + L * a_sigma);
            
            if(trend_linear == 0){          
                Q = matern_52_mat(dx,dx_sq,s_b0[1]/1000,l_b0[1]);
                for(i in 1:nx)
                    Q[i,i] += delta;
                L = cholesky_decompose(Q);
                bi0 = L * to_matrix(a_b0,nx,no);
            }
            
            if(xi_spatial){
                Q = matern_52_mat(dx,dx_sq,s_xi[1],l_xi[1]);
                for(i in 1:nx)
                    Q[i,i] += delta;
                L = cholesky_decompose(Q);
                xi = L * a_xi;
            }
            else if(is_gumbel == 0)
                xi[1] = s_xi[1] * a_xi[1];
            
        }
        else if(nngp == 1){
            b10 = xp * beta_b1 + nngp_chol_vec(dx1,dx2,dx1_sq,dx2_sq,s_b10,l_b10,indM,M,nx,a_b10,delta);
            b00 = nngp_chol_vec(dx1,dx2,dx1_sq,dx2_sq,s_b00/scale_b00,l_b00,indM,M,nx,a_b00,delta);
            sigma = exp(xp * beta_s + nngp_chol_vec(dx1,dx2,dx1_sq,dx2_sq,s_sigma,l_sigma,indM,M,nx,a_sigma,delta));
            
            if(trend_linear == 0)
                bi0 = nngp_chol_mat(dx1,dx2,dx1_sq,dx2_sq,s_b0[1]/1000,l_b0[1],indM,M,nx,to_matrix(a_b0,no,nx,0),delta);
            
            if(xi_spatial)
                xi = nngp_chol_vec(dx1,dx2,dx1_sq,dx2_sq,s_xi[1],l_xi[1],indM,M,nx,a_xi,delta);
            else if(is_gumbel == 0)
                xi[1] = s_xi[1] * a_xi[1];
            
        }
        
        if(trend_linear){   
            b1 = to_vector(b00 * t + rep_matrix(b10,no));
        }
        else{
            // spatiotemporal integrated random walk for the GEV location parameter
            vector[nx] b1_tmp;
            vector[nx] b0_tmp;
            int ct;
            b0[:,1] = b00 + bi0[:,1];
            b1[1:nx] = b10 + b00;
            ct = 1;
            for(i in 2:no){
                b0_tmp = b0[:,i-1];
                b1_tmp = b1[ct:(ct+nx-1)];
                ct += nx;
                b0[:,i] = b0_tmp + bi0[:,i];
                b1[ct:(ct+nx-1)] = b1_tmp + b0_tmp;
            }
        }
        
        {    
            matrix[nli,nx] wl;
            matrix[no,nx] A_wl;
                              
            wl = kernels(dl,tau,row_ones);
            wl = exp(log(wl)*(1/alpha));
            if(nnknot == 0)
                A_wl = A * wl;
            else{
                for(i in 1:nx)
                    A_wl[:,i] = A[:,indMk[i,:]] * wl[:,i];
            }
            
            if(is_gumbel == 0){
                real tmp;
                vector[nx*no] theta;
                vector[nx * no* xi_spatial] xir;
                vector[nx*no] s_bound = rep_vector(0,nx*no);
                vector[nx*no] sigmar;
                int ct = 1;
                
                xi_star = xi * alpha;
                
                if(xi_spatial){
                    xir = to_vector(rep_matrix(xi,no));
                    s_bound[kok] = (b1[kok] - y_ok) .* xir[kok]; // Constraint on sigma
                    
                    for(i in 1:no){
                        for(j in 1:nx){
                            theta[ct] = pow(A_wl[i,j],xi_star[j]);
                            ct += 1;
                        }
                    }
                    
                }
                else{
                    real xi_star_ = xi_star[1];
                    s_bound[kok] = (b1[kok] - y_ok) * xi[1]; // Constraint sigma
                    
                    for(i in 1:no){
                        for(j in 1:nx){
                            theta[ct] = pow(A_wl[i,j],xi_star_);
                            ct += 1;
                        }
                    }
                }

                for(i in 1:nx){
                    tmp = max(s_bound[indt[i,:]]);
                    if(tmp>0)
                        sigma[i] += tmp; // enforce GEV support through constraint on sigma
                }
                sigmar = to_vector(rep_matrix(sigma,no));
                
                if(xi_spatial)
                    mu_star = b1 + (sigmar ./ xir) .* (theta - 1);
                else
                    mu_star = b1 + (sigmar / xi[1]) .* (theta - 1);
                    
                sigma_star = alpha * sigmar .* theta;
            }
            else{
                sigma_star = alpha * to_vector(rep_matrix(sigma,no));
                mu_star = b1 + sigma_star .* log(to_vector(A_wl'));
            }
        }
        
    }
    model {
        alpha_raw ~ uniform(0,5);
        
        // priors on std parameters
        s_b10 ~ std_normal();
        s_b00 ~ std_normal();
        s_sigma ~ std_normal();
        
        // priors on length scale parameters
        l_b10 ~ normal(0,0.5);
        l_b00 ~ normal(0,0.5);
        l_sigma ~ normal(0,0.5);
        tau ~ normal(0,0.5);
        if(xi_spatial)
            l_xi ~ normal(0,0.5);
        
        // priors on regression coefficients
        beta_b1 ~ normal(0,10);
        beta_s ~ normal(0,10);
        
        if(trend_linear == 0){
            s_b0 ~ std_normal();
            l_b0 ~ normal(0,0.5);
            a_b0 ~ std_normal();    
        }
        
        a_b10 ~ std_normal();
        a_b00 ~ std_normal();
        a_sigma ~ std_normal();
        
        append_col(logA,B) ~ logPS(alpha,no,nl,scale_PS);
        
        if(is_gumbel == 0){
            s_xi ~ normal(0,0.2);
            a_xi ~ std_normal();
            if(xi_spatial)
                y_ok ~ gev_xi(mu_star[kok],sigma_star[kok],to_vector(rep_matrix(xi_star,no))[kok]); // likelihood
            else
                y_ok ~ gev(mu_star[kok],sigma_star[kok],xi_star[1]); // likelihood  
        }
        else{
            y_ok ~ gumbel(mu_star[kok],sigma_star[kok]); // likelihood
        }
    }
    
    """
    
    sm = pystan.StanModel(model_code=modelCode)
    return sm
   

