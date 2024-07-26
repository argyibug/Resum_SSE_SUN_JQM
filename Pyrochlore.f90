module configuration
!----------------------------------------------!
! Most important parameters and data structures
!----------------------------------------------!
 save

 integer :: lx0       
 integer :: ly0       
 integer :: lz0 
 integer :: lx       
 integer :: ly        
 integer :: lz    
 integer :: nn0    ! number of sites    
 integer :: sitnum
 integer :: nbs    ! number of bonds  
 integer :: nn    ! number of sites  
 integer :: nnn 
 integer :: nb    ! number of bonds           
 integer :: mm    ! total string length (mm=ns*ms)
 integer :: nq
 integer :: nq_all
 integer :: ns    ! number of substrings (time slices)
 !integer :: ms    ! maximum sub-string length
 integer :: ntau
 integer :: tstp  
 integer :: bincounter
 integer :: bnd
 integer :: bondmaker
 integer :: clustersize
 integer, allocatable :: legn(:)
 integer :: basis
 integer :: rep_counter

 real(8) :: chn_num
 real(8) :: woff_test

 integer :: dxl
 integer :: mxl

 integer :: nms1=0
 integer :: nms2=0 
 integer :: nms3=0 
 integer :: nms4=0

 integer :: nms0
 integer :: bcty
 real(8) :: nhe1 
 real(8) :: nhe2 
 real(8) :: mag1
 real(8) :: mag2
 real(8) :: auen
 real(8) :: culg
 real(8) :: addele

 real(8) :: beta  ! inverse temperature
 real(8) :: bet0  ! inverse temperature
 real(8) :: pbAc 
 real(8) :: pbBs 
 real(8) :: pbjp 
 real(8) :: pbCn
 real(8) :: pbnm
 real(8) :: Ac0
 real(8) :: Bs0
 real(8) :: jp0
 real(8) :: hz0
 real(8) :: temp0
 real(8) :: temp1
 real(8) :: ntem
 real(8) :: dtem
 real(8) :: qastep
 real(8) :: lgstep
 real(8) :: Ac1
 real(8) :: Bs1
 real(8) :: jp1
 real(8) :: hz1
 real(8) :: Ac
 real(8) :: Bs
 real(8) :: jp
 real(8) :: hz
 real(8) :: prob ! part of the acceptance probability for adding (removing) operator 
 real(8) :: probbilibili  
 real(8) :: dtau
 real(8) :: sun
 integer :: ltyp_org
 integer :: ltyp_nxt
 integer :: statebw
 integer :: statetu
 integer :: opstidx

 integer :: loopnumt
 integer :: loopnum0
 integer :: loopnum1
 integer :: loopnumper
 integer :: loopnumber
 integer :: loopnummax
 integer :: looporder
 integer :: loopphase
 integer :: dloop

 real(8) :: spinvalue

 integer :: nh
 integer :: opstchange
 integer, allocatable :: state(:,:)  
 integer, allocatable :: curstate(:)
 integer, allocatable :: curphase(:)
 integer, allocatable :: phase2jdg(:,:)  
 integer, allocatable :: nxtphase(:)    
 integer, allocatable :: linktable(:,:)        
 integer, allocatable :: bsites(:,:)  
 integer, allocatable :: opstring(:)  
 integer, allocatable :: tauscale(:)
 integer, allocatable :: oporder(:)  
 integer, allocatable :: opdhase(:)  
 integer, allocatable :: rebootloop(:)
 integer, allocatable :: opstatenum(:,:,:)
 integer :: rebootnum
 integer :: updpath_counter
 integer :: vex2record

 integer, allocatable :: legtyp(:)
 integer, allocatable :: opflip(:) 
 integer, allocatable :: sambon(:) 
 integer, allocatable :: bontab(:) 
 integer, allocatable :: frststateop(:) 
 integer, allocatable :: custstateop(:) 
 integer, allocatable :: laststateop(:) 
 integer, allocatable :: vertexlist(:,:) 
 integer, allocatable :: vertexlist_map(:) 
 real(8), allocatable :: rantau(:)  
 real(8), allocatable :: spin(:)
 real(8), allocatable :: vec(:,:) 
 integer, allocatable :: phase(:) 
 
 integer, allocatable :: loopoprecord(:,:,:)
 integer, allocatable :: loopop(:,:,:)
 integer, allocatable :: strgop(:,:,:)
 integer, allocatable :: nextlp(:,:,:)
 integer, allocatable :: nextst(:,:,:)
 integer, allocatable :: curoplp(:,:)
 integer, allocatable :: chgloopidx(:,:)
 integer, allocatable :: nxtidx(:)
 integer, allocatable :: loopstate(:,:)
 integer, allocatable :: updatestate(:,:)
 integer, allocatable :: updatememory(:,:)
 integer, allocatable :: midlstate(:,:)
 integer, allocatable :: midstring(:,:)
 integer, allocatable :: nxtloopidx(:)
 integer, allocatable :: sugloopidx(:)
 integer, allocatable :: jdgstring(:,:)
 integer, allocatable :: headsort(:,:)
 integer :: chgloopnum
 integer :: curloopidx
 integer :: rltloopidx

 real(8) :: dmweight
 real(8) :: stweight

 integer, allocatable :: vex2weight(:,:,:,:)
 real(8), allocatable :: change(:,:)
 real(8), allocatable :: weight(:,:)
 real(8), allocatable :: fwight(:,:)
 real(8) :: tprob
 real(8) :: sprob
 real(8) :: dprob

 real(8) :: sec_factor
 real(8), allocatable :: for_factor(:,:)
 real(8), allocatable :: for_result(:)

 real(8), parameter :: pi=3.14159265358979323d0

end module configuration

module measurementdata
 !--------------------------------------------!
 !Data we measured
 !--------------------------------------------!
 save

 real(8) :: enrg1=0.d0
 real(8) :: enrg2=0.d0
 real(8) :: amag1=0.d0
 real(8) :: amag2=0.d0
 real(8) :: amag3=0.d0
 real(8) :: amag4=0.d0
 real(8) :: amag5=0.d0
 real(8) :: amag6=0.d0
 real(8) :: amag7=0.d0
 real(8) :: amag8=0.d0
 real(8) :: amag9=0.d0
 real(8) :: dimr1=0.d0
 real(8) :: dimr2=0.d0
 real(8) :: dimr3=0.d0
 real(8) :: dimr4=0.d0
 real(8) :: dimr5=0.d0
 real(8) :: dimr6=0.d0
 real(8), allocatable :: dimerbg(:)
 real(8), allocatable :: bind1(:)
 real(8), allocatable :: bind2(:)
 real(8) :: tmag1=0.d0
 real(8), allocatable :: stiff(:)


 integer :: signal=0d0

 real(8), allocatable :: crr(:,:,:,:)      !correlation function in real space

 integer, allocatable :: qpts(:)    !list of q-points
 integer, allocatable :: tgrd(:)    !grid of time points
 
 real(8), allocatable :: tcor(:,:)    !correlation function in momentum space
 real(8), allocatable :: tcor_real(:,:)    !correlation function in momentum space
 real(8), allocatable :: tcorpm(:,:)    !correlation function in momentum space
 real(8), allocatable :: tcordm(:,:,:)    !correlation function in momentum space
 real(8), allocatable :: tcordz(:,:,:)    !correlation function in momentum space
 real(8), allocatable :: ref(:,:)     ! real part of fourier transform of states
 real(8), allocatable :: imf(:,:)      ! imaginary part of fourier transform of states
 real(8), allocatable :: phi(:,:,:)   ! phase factors for fourier transforms
 real(8), allocatable :: tc(:) 
 
 real(8), allocatable :: PMRS_temp(:,:)
 real(8), allocatable :: PMinRealSpace(:,:)
 real(8), allocatable :: demo_or(:,:)
 real(8), allocatable :: demo_pm(:,:)
 real(8), allocatable :: DZinRealSpace(:,:,:)
 real(8), allocatable :: DCinRealSpace(:,:,:)
 integer :: nc

 integer :: nod

 character(10) :: resname
 character(10) :: vecname
 character(10) :: bnmname
 character(10) :: bndname
 character(10) :: odpname
 character(10) :: crrname
 character(10) :: crdname
 character(11) :: tcorname
 character(11) :: tcpmname
 character(11) :: td11name
 character(11) :: tz11name
 character(11) :: td12name
 character(11) :: tz12name
 character(11) :: td21name
 character(11) :: tz21name
 character(11) :: td22name
 character(11) :: tz22name
 character(11) :: dmbgname
 character(11) :: autoname

end module measurementdata

!==========================!
!main part of the programme!
!==========================!
! nn     =  number of states
! beta   =  inverse temperature
! dtau   =  time-slize widt
! nbin   =  number of bins
! mstps  =  steps per bin for measurements
! istps  =  initial (equlibration) steps
! gtype  =  time-grid type (1,2,3)
! tmax   =  maximum delta-time for correlation functions
! nqx(y) =  number of q-points for time correlations
! tstp   =  time-step when computing time averages of correlations  
! tmsr   =  measure time correlations after every tmsr MC sweep
! qpts   =  q-values to do (among values 0,....,nn/2, actual q is this times 2*pi/nn)

module vertexupdate
   save
   integer :: v0
   integer :: OpTy0
   integer :: site0
   integer :: site2
   integer :: ndotz
   integer, allocatable :: ndotc(:)

   integer :: v1
   integer :: v2
   integer :: b
   integer :: op
   integer :: loop_counter
   integer :: counter
   integer, allocatable :: dnmsr_counter(:)

   integer :: Spm_measure_signal
   integer :: OpTy1
   integer :: site1
   integer :: vo
   integer :: mm0
   integer :: taumark_o

   integer :: cc
   integer :: cc0
   integer :: cc1

   real(8) :: tau0
   real(8) :: delta_tau

end module vertexupdate

program SpinIce

   use configuration
   use measurementdata
   implicit none
   include 'mpif.h'
   integer :: ierr,nprocs,i1,i2,i3,mctime
   integer :: rank,sz,numm,bsteps
   integer :: i,j,nbins,msteps,isteps,tmsr,gtype
   real(8) :: tmax,qq,slprob
   integer :: njp,nhz
   real(8) :: dhz,djp

   open(unit=10, file='read.in', status="old")
   read(10,*)lx0,ly0,lz0,bcty
   read(10,*)Ac0,nnn
   read(10,*)slprob
   read(10,*)Ac1
   read(10,*)qastep,lgstep
   read(10,*)nbins,msteps,isteps
   read(10,*)dtau
   read(10,*)gtype,tmax              
   read(10,*)tstp,tmsr
   close(unit=10)

   Bs0=0d0
   Bs1=0d0
   bondmaker=1

   open(unit=10, file='temp.in', status="old")
   read(10,*)temp0
   read(10,*)temp1
   read(10,*)ntem
   close(unit=10)
   dtem=(temp1-temp0)/dble(ntem)
   bet0=temp1
   if (temp0==temp1) then
      beta=temp1
      ntem=0d0
   endif

   open(unit=10, file='jp.in', status="old")
   read(10,*)jp0
   read(10,*)jp1
   read(10,*)djp
   close(unit=10)
   njp=int((jp1-jp0)/dble(djp))
   if (jp0==jp1) then
      jp=jp0
      njp=0d0
   endif

   open(unit=10, file='hz.in', status="old")
   read(10,*)hz0
   read(10,*)hz1
   read(10,*)dhz
   close(unit=10)
   nhz=int((hz1-hz0)/dble(dhz))
   if (hz0==hz1) then
      hz=hz0
      nhz=0d0
   endif
   
   sun=2
   basis=0d0

   lx=lx0
   ly=ly0
   lz=lz0
   nn0=4*lx0*ly0*lz0
   beta=bet0

   Ac=Ac0
   jp=jp0
   hz=hz0

   nq=lx0
   open(unit=10, file='q_resolution.in', status="replace")
   close(unit=10)
   open(unit=10, file='q_resolution.in', status="old")
   write(10,*)nq
   close(unit=10)
   nq_all=2*nq+1

   !habs=0d0
   allocate(qpts(nq_all))

   call MPI_init(ierr)
   call MPI_Comm_size(MPI_COMM_WORLD,nprocs,ierr)
   call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

   call initran(1,rank)
   call makelattice()
   call initconfig(rank)
   call taugrid(gtype,tmax)
   call qarrays()

   autoname='auto000.dat'
   i3=rank/100
   i2=mod(rank,100)/10
   i1=mod(rank,10)

   autoname(7:7)=achar(48+i1)
   autoname(6:6)=achar(48+i2)
   autoname(5:5)=achar(48+i3)

   open(10,file=autoname,status='replace')
   close(10)

   print*,"step 1",rank

   prob=beta*pbnm
   numm=int(nn)
   mctime=0d0

!==================================================================!
   !Do isteps equilibration sweeps, find the necessary sweep
   do i=1,isteps
      !print*,"a0",i,loopnummax,loopnumber,-(dble(nh/beta)-pbAc*addele)/nn,autoname
      call Bilibiliupdate(rank,0,numm)
      !print*,"a1",i,loopnummax,loopnumber,-(dble(nh/beta)-pbAc*addele)/nn
      call AcFunupdate(rank,1)
      !print*,"a2",i,loopnummax,loopnumber,-(dble(nh/beta)-pbAc*addele)/nn
      call woffupdate(rank)
      call adjustcutoff(i)
      !print*,"b1",i
      if (mod(i,2500)==1d0) print*,rank,"warm",i,"/",isteps,nh
      !print*,"c",i
      open(10,file=autoname,position='append')
      mctime=mctime+1
      write(10,*)mctime,-(dble(nh/beta)-pbAc*addele)/nn,loopnumber,loopnummax
      close(10)
   enddo
!==================================================================!

   print*,"step 2",rank


   call writeconfig(rank)
   bincounter=0

   print*,"step 3",rank

   ! Do nbins bins with msteps MC sweeps in each, measure after each
   do j=1,nbins
      bincounter=bincounter+1
      do i=1,msteps
         call Bilibiliupdate(rank,0,numm)
         call AcFunupdate(rank,1)
         call woffupdate(rank)
         !print*,"a2",i,nh
         call measure()
         !print*,"b2",i,nh
         if (mod(i,tmsr)==0) then
         endif
         if (mod(i,1000)==1d0) print*,rank,"states",i,"/",msteps
         !print*,"e",i
         open(10,file=autoname,position='append')
         mctime=mctime+1
         write(10,*)mctime,-(dble(nh/beta)-pbAc*addele)/nn,loopnumber,loopnummax
         close(10)
      enddo
      !write bin averages to 'res.dat'!
      print*,bincounter
      call writeresult(rank)
      call writeconfig(rank)
   enddo
   print*,"step 4",rank

   call MPI_Barrier(MPI_COMM_WORLD,ierr)
   call MPI_Finalize(ierr) 

   bincounter=0
   call deallocateall()

end program SpinIce

!==================================================!

!==================================================!

subroutine initconfig(rank)
   use configuration
   use measurementdata
   implicit none

   integer :: i,j
   integer :: rank
   real(8), external :: ran

   allocate(state(nn,2))
   do i=1,nn
      state(i,1)=2*int(2.*ran())-1
      !state(i,1)=1
      state(i,2)=1
      !print*,i,nn,state(i,1)
   enddo
   spinvalue=(sun-1d0)*0.5d0
   allocate(spin(nn))

   ns=int(beta/dtau+0.1d0)
   !mm=max(4*ns,nn/4)
   mm=10
   allocate(curstate(0:dxl-1))
   allocate(curphase(0:dxl-1))
   allocate(phase2jdg(0:dxl-1,2))
   allocate(nxtphase(0:dxl-1))
   phase2jdg(:,:)=0d0

   allocate(bind1(2))
   bind1(:)=0d0
   allocate(bind2(2))
   bind2(:)=0d0
   tmag1=0d0

!=============================================================================!
   allocate(opstring(0:mm-1))            !according to whether it is a sub-programme
   opstring(:)=0                         !according to whether it is a sub-programme
   allocate(tauscale(0:mm-1))            !according to whether it is a sub-programme
   tauscale(:)=0                         !according to whether it is a sub-programme
   allocate(opdhase(0:mm-1))             !according to whether it is a sub-programme
   opdhase(:)=0                         !according to whether it is a sub-programme
   allocate(vertexlist(0:dxl*mm-1,5))        !according to whether it is a sub-programme
   allocate(vertexlist_map(0:dxl*mm-1))    !according to whether it is a sub-programme
   allocate(rantau(nn))             !according to whether it is a sub-programme
   allocate(oporder(nn))
   nh=0                          !according to whether it is a sub-programme
!=============================================================================!
!=============================================================================!
   !call readconfig(rank)          !according to whether it is a sub-programme
!=============================================================================!  
   allocate(frststateop(nn))
   allocate(custstateop(nn))
   allocate(laststateop(nn))
   allocate(rebootloop(2))
   allocate(chgloopidx(0:dxl-1,2))
   allocate(nxtloopidx(0:dxl-1))
   allocate(nxtidx(dxl))
   allocate(sugloopidx(0:dxl-1))
   allocate(midstring(0:dxl-1,0:5))
   allocate(jdgstring(dxl,0:8))
   allocate(headsort(0:dxl-1,0:5))
   allocate(loopoprecord(nn,nn,0:nn))
   allocate(loopop(nn,nn,0:12))
   allocate(loopstate(nn,0:2))
   allocate(updatestate(nn,2))
   allocate(updatememory(nn,2))
   allocate(midlstate(dxl,0:2))
   allocate(strgop(dxl,nn,0:12))
   allocate(nextlp(dxl,nn,0:12))
   allocate(nextst(dxl,dxl,0:12))
   allocate(curoplp(nn,0:dxl))
   allocate(opstatenum(0:4,0:4,6))
   loopoprecord(:,:,:)=0d0
   loopop(:,:,:)=0d0
   loopstate(:,:)=0d0
   midlstate(:,:)=0d0
   strgop(:,:,:)=0d0
   nextlp(:,:,:)=0d0
   nextst(:,:,:)=0d0
   curoplp(:,:)=0d0
   chgloopidx(:,:)=-1
   nxtloopidx(:)=0d0
   sugloopidx(:)=0d0
   midstring(:,:)=0d0
   jdgstring(:,:)=-1d0
   oporder(:)=-1
   frststateop(:)=-1
   custstateop(:)=-1
   laststateop(:)=-1
   rebootloop(:)=0d0
   rebootnum=0d0
   allocate(crr(0:lx0-1,0:ly0-1,0:lz0-1,4))
   crr(:,:,:,:)=0
   allocate(stiff(0:3))
   stiff(:)=0

   do i=1,nn
      !loopstate(i,0)=(int(ran()*2d0)*2-1d0)
      loopstate(i,0)=1d0
      loopstate(i,1)=0d0
   enddo

    vertexlist(:,:)=0
    vertexlist_map(:)=-1
    loopnum0=nn
    loopnum1=0d0
    loopnummax=nn
    loopnumber=loopnum0+loopnum1

   sec_factor=2*(sun*2-1)/sun
   allocate(for_factor(4,2))
   allocate(for_result(2))
   !for_factor(1,1)=1d0/9d0*(spinvalue**2)*(spinvalue+1d0)**2
   !for_factor(2,1)=1d0/9d0*(spinvalue**2)*(spinvalue+1d0)**2
   !for_factor(3,1)=1d0/15d0*(spinvalue)*(3d0*spinvalue**3d0+6d0*spinvalue**2+2d0*spinvalue-1d0)
   !for_factor(4,1)=1d0/15d0*(spinvalue)*(3d0*spinvalue**3d0+6d0*spinvalue**2+2d0*spinvalue-1d0)

   !for_factor(1,2)=0d0
   !for_factor(2,2)=2d0/9d0*(spinvalue**2)*(spinvalue+1d0)**2
   !for_factor(3,2)=1d0/120d0*((2*spinvalue+1)**4-1d0)
   !for_factor(4,2)=2d0/15d0*(spinvalue)*(spinvalue**3d0+2d0*spinvalue**2-spinvalue-2d0)

   for_factor(1,1)=4*(1-1d0/sun)**2
   for_factor(2,1)=4*(1-1d0/sun)**2
   for_factor(3,1)=4*(1-1d0/sun)**2
   for_factor(4,1)=4*(1-1d0/sun)**2

   for_factor(1,2)=0d0
   for_factor(2,2)=4*(1-1d0/sun)**2*sun
   for_factor(3,2)=4*(1-1d0/sun)**2*sun**2
   for_factor(4,2)=-4*(sun-1)/dble(sun)
   
   !for_factor(1,2)=0
   !for_factor(2,2)=0
   !for_factor(3,2)=0
   !for_factor(4,2)=1

   allocate(change(0:4,0:4))
   allocate(weight(0:4,0:4))
   allocate(fwight(0:4,0:4))
   allocate(vex2weight(0:12,0:1,0:1,2))
   call initweight()

   chn_num=sun+1

end subroutine initconfig
!==================================================!

subroutine makelattice()
   use configuration
   implicit none
   integer :: s,x1,x2,y1,y2,z1,z2,index,i
   integer :: s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,bt,bc
   integer :: s11,s12,s13,s14,s15,s16,s17,s18,s19,s20
   integer :: s21,s22,s23,s24,s25,bmk
   integer :: xx1,xx2,xx3,xx4,yy1,yy2,yy3,yy4
   integer :: numbd

   !bondmaker=1
   !bondmaker=0

   allocate(legn(4))

   nn=lx*ly*lz
   print*,nn,lx,ly,lz
   bc=0d0
   pbAc=0d0
   pbBs=0d0
   pbCn=0d0

   if (lz==1) then
      if (ly==1) then
         nb=lx
         mxl=2
         dxl=2*mxl
         allocate(bsites(2,nb))
         allocate(sambon(nb))
         sambon(:)=-1d0
         allocate(bontab(nb))
         bontab(:)=-1d0
         bc=0d0
         do x1=0,lx-1
            s=x1+1
            x2=mod(x1+1,lx)
            bsites(1,s)=s
            bsites(2,s)=x2+1
            bc=bc+1
            sambon(bc)=s
            bontab(s)=bc
         enddo

      elseif (ly==2) then
         allocate(bsites(2,nb))
         allocate(sambon(nb))
         sambon(:)=-1d0
         allocate(bontab(nb))
         bontab(:)=-1d0
         do y1=0,ly-1
            do x1=0,lx-1
               s=1+x1+y1*lx
               x2=mod(x1+1,lx)
               y2=y1
               bsites(1,s)=s
               bsites(2,s)=1+x2+y2*lx
               bc=bc+1
               sambon(bc)=s
               bontab(s)=bc
               if (y1==0) then
                  x2=x1
                  y2=mod(y1+1,ly)
                  bsites(1,s+nn)=s
                  bsites(2,s+nn)=1+x2+y2*lx 
                  bc=bc+1
                  sambon(bc)=s+nn   
                  bontab(s+nn)=bc   
               endif
            enddo
         enddo

      else
         nb=2*nn
         !allocate(bsites(mxl,nb))
         !do y1=0,ly-1
         !   do x1=0,lx-1
         !      s=1+x1+y1*lx
         !
         !      s1=1+mod(x1,lx)+mod(y1,ly)*lx
         !      s2=1+mod(x1-1+lx,lx)+mod(y1,ly)*lx
         !      s3=1+mod(x1,lx)+mod(y1-1+ly,ly)*lx
         !
         !      bsites(1,s)=s1
         !      bsites(2,s)=s3+nn
         !      bsites(3,s)=s2
         !      bsites(4,s)=s1+nn
         !      bsites(5:mxl,s)=-1 
         !
         !      s1=1+mod(x1,lx)+mod(y1,ly)*lx
         !      s2=1+mod(x1+1,lx)+mod(y1,ly)*lx
         !      s3=1+mod(x1,lx)+mod(y1+1,ly)*lx
         !
         !      bsites(1,s+nn)=s1
         !      bsites(2,s+nn)=s1+nn
         !      bsites(3,s+nn)=s3
         !      bsites(4,s+nn)=s2+nn
         !      bsites(5:mxl,s+nn)=-1  
         !   enddo
         !enddo

         mxl=5
         dxl=2*mxl
         allocate(bsites(mxl,nb))
         allocate(sambon(nb))
         sambon(:)=-1d0
         allocate(bontab(nb))
         bontab(:)=-1d0
         do y1=0,ly-1
            do x1=0,lx-1
               s=1+x1+y1*lx

               xx1=x1
               yy1=y1
               xx2=x1+1
               yy2=y1
               xx3=x1+1
               yy3=y1+1
               xx4=x1
               yy4=y1+1

               s1=1+mod(xx1+lx,lx)+mod(yy1+ly,ly)*lx
               s2=1+mod(xx2+lx,lx)+mod(yy2+ly,ly)*lx
               s3=1+mod(xx3+lx,lx)+mod(yy3+ly,ly)*lx
               s4=1+mod(xx4+lx,lx)+mod(yy4+ly,ly)*lx
               bt=0
               
               if (bcty==0d0) then
               elseif (bcty==1d0) then
                  if (0<=xx1 .and. xx1<lx) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx2 .and. xx2<lx) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx3 .and. xx3<lx) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx4 .and. xx4<lx) then 
                  else
                     bt=-1
                  endif
               elseif (bcty==2d0) then
                  if (0<=xx1 .and. xx1<lx .and. 0<=yy1 .and. yy1<ly) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx2 .and. xx2<lx .and. 0<=yy2 .and. yy2<ly) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx3 .and. xx3<lx .and. 0<=yy3 .and. yy3<ly) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx4 .and. xx4<lx .and. 0<=yy4 .and. yy4<ly) then 
                  else
                     bt=-1
                  endif
               endif

               bsites(1,s)=s1
               bsites(2,s)=s2
               bsites(3,s)=s3
               bsites(4,s)=s4
               bsites(5,s)=s1+nn
               if (bt==0d0) then
                  bc=bc+1
                  sambon(bc)=s
                  bontab(s+nn)=bc
                  pbAc=pbAc+1
               endif   

               xx1=x1
               yy1=y1
               xx2=x1-1
               yy2=y1-1
               xx3=x1
               yy3=y1-1
               xx4=x1-1
               yy4=y1

               s1=1+mod(xx1+lx,lx)+mod(yy1+ly,ly)*lx
               s2=1+mod(xx2+lx,lx)+mod(yy2+ly,ly)*lx
               s3=1+mod(xx3+lx,lx)+mod(yy3+ly,ly)*lx
               s4=1+mod(xx4+lx,lx)+mod(yy4+ly,ly)*lx
               bt=0
               
               if (bcty==0d0) then
               elseif (bcty==1d0) then
                  if (0<=xx1 .and. xx1<lx) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx2 .and. xx2<lx) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx3 .and. xx3<lx) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx4 .and. xx4<lx) then 
                  else
                     bt=-1
                  endif
               elseif (bcty==2d0) then
                  if (0<=xx1 .and. xx1<lx .and. 0<=yy1 .and. yy1<ly) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx2 .and. xx2<lx .and. 0<=yy2 .and. yy2<ly) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx3 .and. xx3<lx .and. 0<=yy3 .and. yy3<ly) then 
                  else
                     bt=-1
                  endif
                  if (0<=xx4 .and. xx4<lx .and. 0<=yy4 .and. yy4<ly) then 
                  else
                     bt=-1
                  endif
               endif
         
               bsites(1,s+nn)=s2+nn
               bsites(2,s+nn)=s3+nn
               bsites(3,s+nn)=s1+nn
               bsites(4,s+nn)=s4+nn
               bsites(5,s+nn)=s1
               if (bt==0d0) then
                  bc=bc+1
                  sambon(bc)=s+nn
                  bontab(s)=bc
                  pbBs=pbBs+1
               endif  
            enddo
         enddo
         sitnum=nn
         nn=2*nn
      endif
      !do i=0,ly-1
      !   s1=1+0+i*lx
      !   s2=1+lx-1+i*lx
      !   print*,bontab(s1:s2)
      !enddo
      !do i=0,ly-1
      !   s1=1+0+i*lx
      !   s2=1+lx-1+i*lx
      !   print*,bontab(s1+sitnum:s2+sitnum)
      !enddo
      !print*,sambon(:),bc
      !pause
   else
      nb=12*nn+24*nn+36*nn
      mxl=2
      numbd=12-1
      dxl=2*mxl

      allocate(bsites(mxl,nb))
      allocate(sambon(nb))
      sambon(:)=-1d0
      allocate(bontab(nb))
      bontab(:)=-1d0
      bsites(:,:)=-1
      do x1=0,lx-1
         do y1=0,ly-1
            do z1=0,lz-1
               s=1+x1+y1*lx+z1*lx*ly

               s1=1+mod(x1+0+lx,lx)+mod(y1+0+ly,ly)*lx+mod(z1+0+lz,lz)*lx*ly
               s2=1+mod(x1+0+lx,lx)+mod(y1+0+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly
               s3=1+mod(x1-1+lx,lx)+mod(y1+0+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly
               s4=1+mod(x1+0+lx,lx)+mod(y1-1+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly

               s5=1+mod(x1+1+lx,lx)+mod(y1+0+ly,ly)*lx+mod(z1+0+lz,lz)*lx*ly
               s6=1+mod(x1+0+lx,lx)+mod(y1+1+ly,ly)*lx+mod(z1+0+lz,lz)*lx*ly
               s7=1+mod(x1-1+lx,lx)+mod(y1+0+ly,ly)*lx+mod(z1+0+lz,lz)*lx*ly
               s8=1+mod(x1+0+lx,lx)+mod(y1-1+ly,ly)*lx+mod(z1+0+lz,lz)*lx*ly

               s9=1+mod(x1+0+lx,lx)+mod(y1+0+ly,ly)*lx+mod(z1-1+lz,lz)*lx*ly
               s10=1+mod(x1+1+lx,lx)+mod(y1-1+ly,ly)*lx+mod(z1+0+lz,lz)*lx*ly
               s11=1+mod(x1+0+lx,lx)+mod(y1+1+ly,ly)*lx+mod(z1-1+lz,lz)*lx*ly
               s12=1+mod(x1-1+lx,lx)+mod(y1+1+ly,ly)*lx+mod(z1+0+lz,lz)*lx*ly

               s13=1+mod(x1+1+lx,lx)+mod(y1+0+ly,ly)*lx+mod(z1-1+lz,lz)*lx*ly
               s14=1+mod(x1-1+lx,lx)+mod(y1-1+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly
               s15=1+mod(x1-2+lx,lx)+mod(y1+0+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly
               s16=1+mod(x1+0+lx,lx)+mod(y1+1+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly

               s17=1+mod(x1+0+lx,lx)+mod(y1-2+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly
               s18=1+mod(x1-1+lx,lx)+mod(y1+1+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly
               s19=1+mod(x1-2+lx,lx)+mod(y1+1+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly
               s20=1+mod(x1+1+lx,lx)+mod(y1+0+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly

               s21=1+mod(x1+1+lx,lx)+mod(y1-1+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly
               s22=1+mod(x1+1+lx,lx)+mod(y1-2+ly,ly)*lx+mod(z1+1+lz,lz)*lx*ly
               s23=1+mod(x1-1+lx,lx)+mod(y1+2+ly,ly)*lx+mod(z1+2+lz,lz)*lx*ly
               s24=1+mod(x1-1+lx,lx)+mod(y1-1+ly,ly)*lx+mod(z1+2+lz,lz)*lx*ly
               s25=1+mod(x1-1+lx,lx)+mod(y1+0+ly,ly)*lx+mod(z1+2+lz,lz)*lx*ly

               bmk=0d0
               bsites(1,s)=s1
               bsites(2,s)=s1+nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1

               bmk=1d0
               bsites(1,s+bmk*nn)=s1+nn
               bsites(2,s+bmk*nn)=s1+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1

               bmk=2d0
               bsites(1,s+bmk*nn)=s1+2*nn
               bsites(2,s+bmk*nn)=s1
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1
               
               bmk=3d0
               bsites(1,s+bmk*nn)=s1
               bsites(2,s+bmk*nn)=s1+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1

               bmk=4d0
               bsites(1,s+bmk*nn)=s1+nn
               bsites(2,s+bmk*nn)=s1+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1

               bmk=5d0
               bsites(1,s+bmk*nn)=s1+2*nn
               bsites(2,s+bmk*nn)=s1+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1

               bmk=6d0
               bsites(1,s+bmk*nn)=s3+nn
               bsites(2,s+bmk*nn)=s4+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1

               bmk=7d0
               bsites(1,s+bmk*nn)=s4+2*nn
               bsites(2,s+bmk*nn)=s2
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1

               bmk=8d0
               bsites(1,s+bmk*nn)=s2
               bsites(2,s+bmk*nn)=s3+nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1
               
               bmk=9d0
               bsites(1,s+bmk*nn)=s3+nn
               bsites(2,s+bmk*nn)=s1+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1
            
               bmk=10d0
               bsites(1,s+bmk*nn)=s4+2*nn
               bsites(2,s+bmk*nn)=s1+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1

               bmk=11d0
               bsites(1,s+bmk*nn)=s2
               bsites(2,s+bmk*nn)=s1+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbAc=pbAc+1

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

               bmk=numbd+1
               bsites(1,s+bmk*nn)=s1+1*nn
               bsites(2,s+bmk*nn)=s2+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+2
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s5+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+3
               bsites(1,s+bmk*nn)=s1+2*nn
               bsites(2,s+bmk*nn)=s6+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1
!=================================================================

               bmk=numbd+4
               bsites(1,s+bmk*nn)=s1+0*nn
               bsites(2,s+bmk*nn)=s10+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+5
               bsites(1,s+bmk*nn)=s1+2*nn
               bsites(2,s+bmk*nn)=s4+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+6
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s8+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1
!=================================================================

               bmk=numbd+7
               bsites(1,s+bmk*nn)=s1+0*nn
               bsites(2,s+bmk*nn)=s3+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+8
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s12+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+9
               bsites(1,s+bmk*nn)=s1+1*nn
               bsites(2,s+bmk*nn)=s7+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1
!=================================================================

               bmk=numbd+10
               bsites(1,s+bmk*nn)=s1+0*nn
               bsites(2,s+bmk*nn)=s11+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+11
               bsites(1,s+bmk*nn)=s1+1*nn
               bsites(2,s+bmk*nn)=s13+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+12
               bsites(1,s+bmk*nn)=s1+2*nn
               bsites(2,s+bmk*nn)=s9+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

!=================================================================

               bmk=numbd+13
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s2+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+14
               bsites(1,s+bmk*nn)=s3+2*nn
               bsites(2,s+bmk*nn)=s1+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+15
               bsites(1,s+bmk*nn)=s2+0*nn
               bsites(2,s+bmk*nn)=s3+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1
!=================================================================

               bmk=numbd+16
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s4+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+17
               bsites(1,s+bmk*nn)=s2+0*nn
               bsites(2,s+bmk*nn)=s1+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+18
               bsites(1,s+bmk*nn)=s4+1*nn
               bsites(2,s+bmk*nn)=s2+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1
!=================================================================

               bmk=numbd+19
               bsites(1,s+bmk*nn)=s3+2*nn
               bsites(2,s+bmk*nn)=s4+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+20
               bsites(1,s+bmk*nn)=s4+1*nn
               bsites(2,s+bmk*nn)=s1+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+21
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s3+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1
!=================================================================

               bmk=numbd+22
               bsites(1,s+bmk*nn)=s3+2*nn
               bsites(2,s+bmk*nn)=s2+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+23
               bsites(1,s+bmk*nn)=s2+0*nn
               bsites(2,s+bmk*nn)=s4+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

               bmk=numbd+24
               bsites(1,s+bmk*nn)=s4+1*nn
               bsites(2,s+bmk*nn)=s3+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbBs=pbBs+1

!=================================================================
               bmk=numbd+25
               bsites(1,s+bmk*nn)=s1+0*nn
               bsites(2,s+bmk*nn)=s2+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+26
               bsites(1,s+bmk*nn)=s1+1*nn
               bsites(2,s+bmk*nn)=s2+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+27
               bsites(1,s+bmk*nn)=s1+0*nn
               bsites(2,s+bmk*nn)=s5+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+28
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s5+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+29
               bsites(1,s+bmk*nn)=s1+0*nn
               bsites(2,s+bmk*nn)=s6+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+30
               bsites(1,s+bmk*nn)=s1+2*nn
               bsites(2,s+bmk*nn)=s6+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

!========================================================
               bmk=numbd+31
               bsites(1,s+bmk*nn)=s1+1*nn
               bsites(2,s+bmk*nn)=s8+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+32
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s8+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+33
               bsites(1,s+bmk*nn)=s1+1*nn
               bsites(2,s+bmk*nn)=s4+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+34
               bsites(1,s+bmk*nn)=s1+2*nn
               bsites(2,s+bmk*nn)=s4+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+35
               bsites(1,s+bmk*nn)=s1+1*nn
               bsites(2,s+bmk*nn)=s10+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+36
               bsites(1,s+bmk*nn)=s1+0*nn
               bsites(2,s+bmk*nn)=s10+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

!========================================================
               bmk=numbd+37
               bsites(1,s+bmk*nn)=s1+2*nn
               bsites(2,s+bmk*nn)=s3+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+38
               bsites(1,s+bmk*nn)=s1+0*nn
               bsites(2,s+bmk*nn)=s3+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+39
               bsites(1,s+bmk*nn)=s1+2*nn
               bsites(2,s+bmk*nn)=s7+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+40
               bsites(1,s+bmk*nn)=s1+1*nn
               bsites(2,s+bmk*nn)=s7+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+41
               bsites(1,s+bmk*nn)=s1+2*nn
               bsites(2,s+bmk*nn)=s12+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+42
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s12+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

!========================================================
               bmk=numbd+43
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s9+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+44
               bsites(1,s+bmk*nn)=s1+2*nn
               bsites(2,s+bmk*nn)=s9+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+45
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s11+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+46
               bsites(1,s+bmk*nn)=s1+0*nn
               bsites(2,s+bmk*nn)=s11+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+47
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s13+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+48
               bsites(1,s+bmk*nn)=s1+1*nn
               bsites(2,s+bmk*nn)=s13+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

!========================================================
               bmk=numbd+49
               bsites(1,s+bmk*nn)=s2+0*nn
               bsites(2,s+bmk*nn)=s18+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+50
               bsites(1,s+bmk*nn)=s3+2*nn
               bsites(2,s+bmk*nn)=s12+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+51
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s6+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

!========================================================
               bmk=numbd+52
               bsites(1,s+bmk*nn)=s4+1*nn
               bsites(2,s+bmk*nn)=s21+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+53
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s10+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+54
               bsites(1,s+bmk*nn)=s2+0*nn
               bsites(2,s+bmk*nn)=s5+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1
!========================================================
               bmk=numbd+55
               bsites(1,s+bmk*nn)=s3+2*nn
               bsites(2,s+bmk*nn)=s14+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+56
               bsites(1,s+bmk*nn)=s4+1*nn
               bsites(2,s+bmk*nn)=s8+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+57
               bsites(1,s+bmk*nn)=s1+3*nn
               bsites(2,s+bmk*nn)=s7+3*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1
!========================================================
               bmk=numbd+58
               bsites(1,s+bmk*nn)=s4+1*nn
               bsites(2,s+bmk*nn)=s24+1*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+59
               bsites(1,s+bmk*nn)=s3+2*nn
               bsites(2,s+bmk*nn)=s25+2*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

               bmk=numbd+60
               bsites(1,s+bmk*nn)=s2+0*nn
               bsites(2,s+bmk*nn)=s23+0*nn
               bc=bc+1
               sambon(bc)=s+bmk*nn
               bontab(s+bmk*nn)=bc
               pbCn=pbCn+1

            enddo
         enddo
      enddo
      sitnum=nn
      nn=4*nn
   endif
   nbs=bc
   print*,nbs,nb,nn,sitnum,pbAc,pbBs,pbCn
   !pause

   if (bondmaker==1) then
      legn(1)=2
      legn(2)=2
      legn(3)=1
      legn(4)=2
   elseif (bondmaker==0) then
      legn(1)=2
      legn(2)=2
      legn(3)=1
      legn(4)=2
   endif

   !s2=0d0
   !s=1+x1+y1*lx+z1*lx*ly
   !do s1=1,nn/3
      !print*,int((s1-1)/(lx*ly)),s2
      !if (int((s1-1)/(lx*ly))==s2) then
         !print*,s1,s2
         !print*,bsites(9:mxl,s1)
         !print*,bsites(5:8,s1)
         !print*,bsites(1:4,s1)
         !print*,"//////////////////////"
         !print*,bsites(9:mxl,s1+nn/3)
         !print*,bsites(5:8,s1+nn/3)
         !print*,bsites(1:4,s1+nn/3)
         !print*,"++++++++++++++++++++++++"
         !print*,bsites(9:mxl,s1+2*nn/3)
         !print*,bsites(5:8,s1+2*nn/3)
         !print*,bsites(1:4,s1+2*nn/3)
         !print*,"++++++++++++++++++++++++"
         !print*,bsites(9:mxl,s1+3*nn/3)
         !print*,bsites(5:8,s1+3*nn/3)
         !print*,bsites(1:4,s1+3*nn/3)
         !print*,"//////////////////////"
      !endif
      !if (mod((s1-1),(lx*ly))==lx*ly-1) s2=s2+1
   !enddo


   pbjp=0d0
   if (nnn==1) then
      pbnm=pbAc+pbBs+pbjp+pbCn
      nbs=nbs
   else
      pbnm=pbAc+pbjp
      nbs=pbAc
   endif

   call makelinktype()

end subroutine makelattice
!==================================================!
!==================================================!

subroutine makelinktype()
   use configuration
   implicit none
   integer :: s,x1,x2,y1,y2,s1,s2,s3,s4,s5,s6,t,i

   allocate(linktable(0:dxl-1,0:2))

   do i=0,dxl-1
      linktable(i,0)=mod(i+mxl,dxl)
      linktable(i,1)=ieor(i,1)
      linktable(i,2)=ieor(mod(i+mxl,dxl),1)
   enddo
   !do i=0,dxl-1
   !  print*,"0",i,linktable(i,0)
   !enddo
   !do i=0,dxl-1
   !  print*,"1",i,linktable(i,1)
   !enddo
   !do i=0,dxl-1
   !  print*,"2",i,linktable(i,2)
   !enddo
   !pause

end subroutine makelinktype
!==================================================!

subroutine Bilibiliupdate(rank,update_type,lim_counter)
   use configuration
   use measurementdata
   implicit none

   integer :: i,j,b,s,rank,cs,s1,k,sig,crsign,update_type,lim_counter,update_counter,mm_sug_counter
   integer :: mm_sug,mm_tmp,opod_sug,opod_pre,opod_aft,idx_sug,num_sug,mm_next,nh_tmp,op
   integer :: vtmp
   real(8) :: wght1,wght2,wght3,wghtq3
   integer, allocatable :: tmststateop(:)
   integer, allocatable :: opstridx(:)
   integer, allocatable :: mm_sug_tab(:)
   integer, allocatable :: site(:)
   real(8), external :: ran
   real(8), external :: hf
   allocate(tmststateop(nn))
   allocate(site(0:dxl))
   tmststateop(:)=0d0

   !wght1=g2*dble(1d0/sun)
   !wght2=jq1*dble(1d0/sun)
   !wght3=0d0
   !wghtq3=qq3*(dble(1d0/sun)**bnd)
   wght1=dble(0.5d0)
   wght2=dble(0.5d0)
   wght3=0d0
   wghtq3=(dble(0.5d0)**bnd)
   update_counter=0d0
   i=mm-1
   updpath_counter=0d0
   !do i=0,mm-1
   !mm_sug=int(mm*ran())

   if (update_type==0d0) then
   elseif (update_type==1d0) then
   elseif (update_type==2d0) then
      allocate(opstridx(0:mm-1))
      opstridx(:)=0
      mm_sug_counter=0d0
      idx_sug=1
   endif

   !print*,"======================================"
   !tmststateop(:)=custstateop(:)
   !do j=1,nh
   !  mm_tmp=oporder(j)
   !  print*,j,mm_tmp,nh
   !  if (opstring(mm_tmp)==0) pause
   !  b=opstring(mm_tmp)/4
   !  call gencurstateop(mm_tmp,b,0)
   !enddo
   !print*,nh+1,oporder(nh+1),nh

   !do j=1,nn
   !  if (custstateop(j)/=tmststateop(j)) then
   !     print*,"miss"
   !  else
   !  !print*,"check"
   !  endif
   !enddo
   !print*,"0000000000000000000000000000000000"

   do
      if (update_type==0) then
         i=mod(i+1,mm)
         if (update_counter>=mm) then
            exit
         endif
      elseif (update_type==1) then
         mm_sug=int(mm*ran())
         opod_sug=tauscale(mm_sug)

         if (update_counter>lim_counter) then
            exit
         endif
         !print*,mm_sug,opod_sug,opstring(mm_sug),nh
         if (nh==0d0) then
         else
            if (opstring(mm_sug)==0) then
               opod_pre=mod(opod_sug+nh-1,nh)+1
            else
               opod_pre=mod(opod_sug+nh-2,nh)+1
            endif
            call preupdate(rank,mm_sug,1,opod_pre)
         endif

         i=mm_sug
      elseif (update_type==2) then
         !print*,"idx_sug",idx_sug,num_sug
         !print*,updpath_counter,num_sug,updpath_counter>=nh*dxl
         if (idx_sug==1) then
            if (mm_sug_counter>1 .and. updpath_counter>=2*nn) then
               !print*,updpath_counter,num_sug,updpath_counter>=nh*dxl
               exit
            endif
            num_sug=nn
            !num_sug=max(nn*int(sun),nh)
            num_sug=min(mm,num_sug)
            !print*,num_sug
            allocate(mm_sug_tab(num_sug))
            opstridx(:)=0
            mm_sug_tab(:)=-1
            idx_sug=0d0 
            do
               idx_sug=idx_sug+1
               if (idx_sug>num_sug) exit
               do
                  mm_sug=int(mm*ran())
                  if (opstridx(mm_sug)==0d0) then
                     opstridx(mm_sug)=opstridx(mm_sug)+1d0
                     exit
                  endif
               enddo
               mm_sug_tab(idx_sug)=mm_sug
               !print*,mm_sug,idx_sug
            enddo

            call quick_sort_int(mm_sug_tab,num_sug,1,num_sug)
            opstridx(:)=0d0
            idx_sug=1d0
            nh_tmp=nh
         endif

         mm_sug=mm_sug_tab(idx_sug)
         opod_sug=tauscale(mm_sug)
         if (nh==0d0) then
         else
            if (idx_sug==1) then
               if (opstring(mm_sug)==0) then
                  opod_pre=mod(opod_sug+nh-1,nh)+1
               else
                  opod_pre=mod(opod_sug+nh-2,nh)+1
               endif
               call preupdate(rank,mm_sug,1,opod_pre)
            endif
         endif
         i=mm_sug
      endif

      !print*,"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",mm_sug,opstring(mm_sug),nh
      update_counter=update_counter+1
      !print*,"start",i,update_type,update_counter,lim_counter,mm
!=============================================================================================================================!
      op=opstring(i)
      if ( op==0 ) then
         b=int(pbnm*ran())+1
         do j=0,mxl-1
            site(j)=bsites(j+1,b)
            if (custstateop(site(j))>=0) then
               curstate(j)=loopstate(vertexlist(custstateop(site(j)),1),0)
               curphase(j)=mod(vertexlist(custstateop(site(j)),5),2)
               curstate(j+mxl)=curstate(j)
               curphase(j+mxl)=curphase(j)
            else
               curstate(j)=state(bsites(j+1,b),1)
               curphase(j)=1
               curstate(j+mxl)=curstate(j)
               curphase(j+mxl)=1
            endif
         enddo

         do j=0,dxl-1
            if (curstate(j)==0d0) then
               print*,"cur error1",curstate(j),state(bsites(j,b),1),site(j),custstateop(site(j))
               print*,"cur error1",vertexlist(custstateop(site(j)),1),loopstate(vertexlist(custstateop(site(j)),1),0)
               pause
            endif
         enddo
         statebw=int((curstate(0)*curphase(0)+curstate(1)*curphase(1)+2)/2d0)
         if (b<=pbAc .and. sambon(bontab(b))<=nbs) then 
            tprob=change(4,statebw)
            !print*,tprob,weight(4,:)
            dprob=1d0
            if (tprob/=0d0) then
            !print*,"---------------------------------------"
               do j=1,3
                  !print*,tprob,change(j,statebw),change(4,statebw),weight(j,statebw)
                  if (change(j,statebw)==0d0) cycle
                  !print*,"1d0/tprob",1d0/tprob
                  if (ran()*(tprob)<=change(j,statebw)) then
                  !if (ran()*(tprob)<=1d0) then
                     !print*,"1d0/tprob",1d0/tprob
                     ltyp_org=0d0
                     ltyp_nxt=j-1
                     !tprob=change(1,statebw)/tprob
                     !tprob=1d0
                     tprob=change(j,statebw)/change(4,statebw)
                     !tprob=change(4,statebw)
                     sprob=weight(j,statebw)
                     !print*,ltyp_org,ltyp_nxt,statebw
                     call caldloop(i,b,1,ltyp_nxt)
                     !print*,"add",tprob
                     !print*,mm,nh
                     if ( ran()*(mm-nh)<=(sprob*prob*tprob) ) then
                        !print*,"add",tprob,sprob,dprob,statebw
                        opstring(i)=4*b+ltyp_nxt
                        nh=nh+1
                        call addoperator(i,b,rank)

                        do k=0,mxl-1
                           site(k)=bsites(k+1,b)
                           if (custstateop(site(k))>=0) then
                              state(site(k),2)=0
                           else
                           endif
                        enddo
                     endif
                     exit
                  else
                     tprob=tprob-change(j,statebw)
                     !tprob=tprob-1
                  endif
               enddo
            !print*,"---------------------------------------"
            endif
         elseif (b>pbAc .and. b<=pbAc+pbBs .and. sambon(bontab(b))<=nbs .and. nnn==1d0) then
         elseif (b>pbAc+pbBs .and. b<=pbAc+pbBs+pbCn .and. sambon(bontab(b))<=nbs .and. nnn==1d0) then
         endif
      elseif ( op/=0 ) then
         b=op/4
         do j=0,dxl-1
            vtmp=dxl*i+j
            curstate(j)=loopstate(vertexlist(vtmp,1),0)
            curphase(j)=vertexlist(vtmp,5)
            if (j<mxl) then
               if (state(bsites(j+1,b),2)==0d0) then
                  state(bsites(j+1,b),1)=curstate(j)*curphase(j)
               endif
            endif
         enddo

         do j=0,dxl-1
            if (curstate(j)==0d0) then
               print*,"cur error2"
               pause
            endif
         enddo
         statetu=int((curstate(0)*curphase(0)+curstate(mxl)*curphase(mxl)+2)/2d0)
         statebw=int((curstate(0)*curphase(0)+curstate(1)*curphase(1)+2)/2d0)
         if (statetu==0 .or. statetu==2) then
         else  
            if (statebw/=1) then
               print*,"error"
               pause
            endif          
            statebw=3d0
         endif
         ltyp_org=mod(op,4)
         if (b<=pbAc .and. sambon(bontab(b))<=nbs) then
            tprob=change(4,statebw)
            if (tprob==0d0) pause
            dprob=1d0
            if (tprob/=0d0) then
               do j=1,3
                  !print*,tprob,change(j,statebw),j,statebw,ltyp_org
                  if (ran()*(tprob)<=change(j,statebw)) then
                     ltyp_nxt=j-1
                     if (ltyp_nxt==ltyp_org) then
                        if (statetu==0 .or. statetu==2) then
                           ltyp_nxt=0d0
                           !tprob=1d0
                           tprob=change(ltyp_org+1,statebw)/change(4,statebw)
                           !tprob=change(4,statebw)
                           sprob=1/weight(ltyp_org+1,statebw)
                           !print*,ltyp_org,ltyp_nxt,statebw
                           call caldloop(i,b,-1,ltyp_nxt)
                           !print*,"rem",tprob
                           !print*,mm,nh,(mm-nh+1),(dprob)
                           if ( ran()*prob*tprob<=(mm-nh+1)*(dprob) ) then
                              !print*,"rem",tprob,sprob,dprob
                              opstring(i)=0
                              nh=nh-1
                              call remoperator(i,b,rank)
                           !   !call gencurstateop(i,b,0)
                           !   !print*,"checkpoint1"
                           else
                              call gencurstateop(i,b,0)
                           !   !print*,"checkpoint2"
                           endif
                        else
                           call gencurstateop(i,b,0)
                        endif
                     else
                        tprob=change(ltyp_nxt+1,statebw)/change(ltyp_org+1,statebw)
                        sprob=weight(ltyp_nxt+1,statebw)/weight(ltyp_org+1,statebw)
                        call caldloop(i,b,0,ltyp_nxt)
                        !print*,"rwr",tprob,sprob,dprob,sprob/tprob
                        if ( ran()*tprob<=(dprob) ) then
                        !   !print*,"rwr",tprob,sprob,dprob
                           opstring(i)=4*b+ltyp_nxt
                           call rwroperator(i,b,rank)
                        !   !call gencurstateop(i,b,0)
                        !   !   do k=0,dxl-1
                        !     vtmp=dxl*i+k
                        !     curstate(k)=loopstate(vertexlist(vtmp,1),0)
                        !     curphase(k)=vertexlist(vtmp,5)
                        !   !   enddo
                        !   statetu=int((curstate(0)*curphase(0)+curstate(mxl)*curphase(mxl)+2)/2d0)
                        !  statebw=int((curstate(0)*curphase(0)+curstate(1)*curphase(1)+2)/2d0)
                        !  print*,statetu,statebw
                        else
                           call gencurstateop(i,b,0)
                           !print*,"checkpoint2"
                        endif      
                     endif
                     exit
                  endif
                  tprob=tprob-change(j,statebw)
               enddo
            else
               call gencurstateop(i,b,0)
            endif

         elseif (b>pbAc .and. b<=pbAc+pbBs .and. sambon(bontab(b))<=nbs .and. nnn==1d0) then
         elseif (b>pbAc+pbBs .and. b<=pbAc+pbBs+pbCn .and. sambon(bontab(b))<=nbs .and. nnn==1d0) then
         endif
       endif
      !print*,"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",mm_sug,opstring(mm_sug),nh
      !print*,"update",update_type
!=============================================================================================================================!

      if (update_type==0) then
      elseif (update_type==1) then
         if (nh==0d0) then
         else
            opod_sug=tauscale(mm_sug)
            opod_aft=mod(opod_sug,nh)+1
            call preupdate(rank,mm_sug,opod_aft,nh)
         endif
      elseif (update_type==2) then
         idx_sug=mod(idx_sug,num_sug)+1
         if (nh==0d0) then
            if (idx_sug==1) then
               deallocate(mm_sug_tab)
               mm_sug_counter=mm_sug_counter+1
            endif
         else
            if (idx_sug/=1) then
               opod_sug=tauscale(mm_sug)
               opod_aft=mod(opod_sug,nh)+1
               mm_sug=mm_sug_tab(idx_sug)
               opod_sug=tauscale(mm_sug)
               if (opstring(mm_sug)==0) then
                  opod_pre=mod(opod_sug+nh-1,nh)+1
               else
                  opod_pre=mod(opod_sug+nh-2,nh)+1
               endif
               call preupdate(rank,mm_sug,opod_aft,opod_pre)
            else
               opod_sug=tauscale(mm_sug)
               opod_aft=mod(opod_sug,nh)+1
               mm_sug=mm_sug_tab(idx_sug)
               opod_sug=tauscale(mm_sug)
               call preupdate(rank,mm_sug,opod_aft,nh)
               deallocate(mm_sug_tab)
               mm_sug_counter=mm_sug_counter+1

               !do i=1,nn
               !  if (frststateop(i)/=-1) then
               !     if (frststateop(i)/=vertexlist_map(custstateop(i))) then
               !        print*,"boundary lost",frststateop(i),custstateop(i),vertexlist_map(custstateop(i)),&
               !        &vertexlist_map(frststateop(i)),i
               !        print*,"boundary lost",int(frststateop(i)/dxl),int(custstateop(i)/dxl),&
               !        &mod(opstring(int(frststateop(i)/dxl)),2),mod(opstring(int(custstateop(i)/dxl)),2)
               !        pause
               !     endif
               !  else
               !     if (frststateop(i)/=custstateop(i)) then
               !        print*,"boundary lost",frststateop(i),custstateop(i),i
               !        pause
               !     endif
               !  endif
               !enddo
            endif
         endif
      endif

      do j=1,nn
         if (state(j,2)==1) state(j,1)=2*int(2.*ran())-1
         !if (state(j,1)/=1) then
         !   print*,"2",state(j,1),j
         !   pause
         !endif
      enddo
      !print*,"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++",mm_sug,opstring(mm_sug),nh
   enddo

   if (loopnumber/=loopnum0+loopnum1) then
      print*,loopnumber,loopnum0,loopnum1,"neq"
      pause
   endif

   if (update_type==0) then
      do j=1,nh
         mm_tmp=oporder(j)
         b=opstring(mm_tmp)/4
         call gencurstateop(mm_tmp,b,0)
      enddo
   else
      do j=1,nh
         mm_tmp=oporder(j)
         b=opstring(mm_tmp)/4
         call gencurstateop(mm_tmp,b,0)
      enddo
   endif

   !do i=1,nn
   !  if (frststateop(i)/=-1) then
   !     if (frststateop(i)/=vertexlist_map(custstateop(i))) then
   !        print*,"boundary lost",frststateop(i),custstateop(i),vertexlist_map(custstateop(i)),&
   !        &vertexlist_map(frststateop(i)),i
   !        print*,"boundary lost",int(frststateop(i)/dxl),int(custstateop(i)/dxl),&
   !        &mod(opstring(int(frststateop(i)/dxl)),2),mod(opstring(int(custstateop(i)/dxl)),2)
   !        pause
   !     endif
   !  else
   !     if (frststateop(i)/=custstateop(i)) then
   !        print*,"boundary lost",frststateop(i),custstateop(i),i
   !        pause
   !     endif
   !  endif
   !enddo

end subroutine Bilibiliupdate
!==================================================!subroutine Bilibiliupdate(rank,update_type,lim_counter)
subroutine AcFunupdate(rank,lim_counter)
   use configuration
   use measurementdata
   implicit none

   integer :: i,j,chgcount,lidx_sug,rank,lim_counter,statebw1,statebw2,phase0,phase1
   integer :: k, widx1,widx2,st1,st2,tim_counter,totalop
   real(8) :: AcFprob
   integer, allocatable :: loopstateupdate(:)
   integer, allocatable :: visitupdate(:,:)
   real(8), external :: ran

   opstatenum(:,:,:)=0d0
   allocate(loopstateupdate(loopnummax))
   loopstateupdate(:)=0d0
   allocate(visitupdate(loopnummax,loopnummax))
   visitupdate(:,:)=0d0

   if (loopnum1/=sum(loopstate(:,1))) then
      print*,"mis",loopnum1,loopnum0,loopnumber,sum(loopstate(:,1))
      pause
   endif
   tim_counter=0d0
   do
      tim_counter=tim_counter+1
      chgcount=0d0
      loopstateupdate(:)=0d0
      visitupdate(:,:)=0d0

      do 
         do
            i=int(ran()*mm)*dxl+int(dxl*ran())
            if (vertexlist_map(i)>=0) then
               lidx_sug=vertexlist(i,1)
               exit
            endif
         enddo
         loopstateupdate(lidx_sug)=loopstateupdate(lidx_sug)+1
         !print*,"lid",lidx_sug,loopstateupdate(lidx_sug),tim_counter
         chgcount=chgcount+1
         if (chgcount>=lim_counter) then
            exit
         endif
      enddo
      !print*,sum(loopstateupdate(:))
      !print*,"1",lidx_sug,tim_counter

      opstatenum(:,:,:)=0d0
      totalop=0d0
      do lidx_sug=1,loopnummax
         if (loopstateupdate(lidx_sug)==0d0) then
            cycle
         else
            phase0=-1
            !print*,"phase0",phase0
            do i=1,loopnummax
               if (loopstateupdate(i)==0d0) then
                  phase1=(1)
               else
                  phase1=(-1)
               endif
               !print*,"phase1",phase1
               if (loopop(lidx_sug,i,0)==0d0 .or. loopstate(i,1)==0d0 .or. visitupdate(lidx_sug,i)/=0) then
                  cycle
               else
                  !print*,lidx_sug,i
                  call opnumchange(lidx_sug,i,1,1,phase0,phase1,0)
                  visitupdate(lidx_sug,i)=visitupdate(lidx_sug,i)+1
                  visitupdate(i,lidx_sug)=visitupdate(i,lidx_sug)+1
               endif
            enddo
         endif
      enddo
      !print*,"--------------------------------------------------"
      AcFprob=1d0
      do i=1,3
         do j=0,3
            !print*,i,j,opstatenum(i,j,:),weight(i,j),(opstatenum(i,j,2)-opstatenum(i,j,1))
            if (opstatenum(i,j,2)/=0d0 .or. opstatenum(i,j,1)/=0d0) then
               totalop=totalop+1
               if (weight(i,j)/=0d0) then
                  if (opstatenum(i,j,2)/=opstatenum(i,j,1)) then
                     !print*,i,j,opstatenum(i,j,2)-opstatenum(i,j,1),opstatenum(i,j,2),opstatenum(i,j,1)
                     !print*,weight(i,j),(weight(i,j)**(opstatenum(i,j,2)-opstatenum(i,j,1)))
                     AcFprob=AcFprob*(weight(i,j)**(opstatenum(i,j,2)-opstatenum(i,j,1)))
                  endif
               else
                  AcFprob=0d0
                  print*,"zero"
                  pause
                  exit
               endif
            endif
         enddo
      enddo
      if (totalop==0d0) then
         do lidx_sug=1,loopnummax
            if (loopstateupdate(lidx_sug)/=0) then
               print*,"lid",lidx_sug,loopstateupdate(lidx_sug)  
            endif
         enddo
         do i=1,3
            do j=0,3
               print*,i,j,opstatenum(i,j,:),weight(i,j),(opstatenum(i,j,2)-opstatenum(i,j,1))
            enddo
         enddo
         pause
      endif
      !print*,"AcFprob",AcFprob
      !if (AcFprob/=1d0) then
      !   print*,"AcFprobhg",AcFprob
      !   pause
      !endif
      if (ran()<=AcFprob) then
         do i=1,loopnummax
            if (loopstate(i,1)==0d0) then
               cycle
            else
               phase1=(-1)**mod(loopstateupdate(i),2)
               loopstate(i,0)=loopstate(i,0)*phase1
            endif
         enddo
      endif
      if (tim_counter>nn) exit
   enddo
   !print*,"AcFinish"

end subroutine AcFunupdate
!==================================================!

!==================================================!

subroutine woffupdate(rank)
   use configuration
   implicit none
   integer :: i,j,counter,lidx_sug
   integer :: clustercounter,rank,totalop
   real(8) :: wolfprob,weightchange
   real(8), external :: ran
   integer, allocatable :: visitupdate(:,:)

   allocate(visitupdate(loopnummax,loopnummax))
   visitupdate(:,:)=0d0

   updatestate(:,:)=0d0
   updatememory(:,:)=0d0  

   clustercounter=0d0
   clustersize=0d0
   opstatenum(:,:,:)=0d0
   do 
      !if (clustercounter>=lx) exit
      if (clustercounter>=nn) exit
      opstatenum(:,:,:)=0d0
      updatestate(:,:)=0d0
      updatememory(:,:)=0d0 
      clustersize=0d0
      updatestate(:,1)=loopstate(:,0)
      updatestate(:,2)=1
      visitupdate(:,:)=0d0
      weightchange=0d0

      do
         i=int(ran()*mm)*dxl+int(dxl*ran())
         if (vertexlist_map(i)>=0) then
            lidx_sug=vertexlist(i,1)
            exit
         endif
      enddo
      updatestate(lidx_sug,2)=-1
      
      !call opnumchange(lidx_sug,lidx_sug,1,1,updatestate(lidx_sug,2),updatestate(lidx_sug,2),0)

      updatememory(lidx_sug,1)=1
      updatememory(lidx_sug,2)=1
      clustersize=clustersize+1
      i=mod(lidx_sug+loopnummax-2,loopnummax)+1
      counter=0d0
      woff_test=1d0
      !print*,"0000000000000000000000000000000000000000000000000000",clustersize
      do
         i=mod(i,loopnummax)+1
         if (loopstate(i,1)==0d0) then
            cycle
         endif
         if (i==lidx_sug) counter=counter+1
         !print*,clustersize,counter,lidx_sug,i
         if (clustersize==0) then
            if (sum(updatememory(:,2))/=0d0) then
               print*,"error",sum(updatememory(:,2))   
            endif    
            !print*,loopnum1,sum(updatememory(:,1))        
            !print*,"+++++++++++++++++++++++++++++++++++++++++++++++++"
            exit
         endif   
         if (updatememory(i,2)==1) then
            updatememory(i,2)=0d0
            clustersize=clustersize-1
            call buildcluster(i)
            if (sum(updatememory(:,2))/=clustersize) then
               print*,"cluster error",sum(updatememory(:,2)),clustersize
               pause
            endif
         endif
      enddo
      wolfprob=1d0
      totalop=0d0
      opstatenum(:,:,1:2)=0d0
      do lidx_sug=1,loopnummax
         if (updatestate(lidx_sug,2)==1d0) then
            cycle
         else
            do i=1,loopnummax
               !print*,"phase1",phase1
               if (loopop(lidx_sug,i,0)==0d0 .or. loopstate(i,1)==0d0 .or. visitupdate(lidx_sug,i)/=0) then
                  cycle
               else
                  !print*,lidx_sug,i
                  call opnumchange(lidx_sug,i,1,1,updatestate(lidx_sug,2),updatestate(i,2),0)
                  visitupdate(lidx_sug,i)=visitupdate(lidx_sug,i)+1
                  visitupdate(i,lidx_sug)=visitupdate(i,lidx_sug)+1
               endif
            enddo
         endif
      enddo
      do i=1,3
         do j=0,3
            if (opstatenum(i,j,2)/=0d0 .or. opstatenum(i,j,1)/=0d0) then
               if ((weight(i,j))/=fwight(i,j)) then
                  if (opstatenum(i,j,2)/=opstatenum(i,j,1)) then
                     wolfprob=wolfprob*((weight(i,j)/fwight(i,j))**(opstatenum(i,j,2)-opstatenum(i,j,1)))
                  endif
               else
                  cycle
                  !print*,"zero"
                  !pause
                  !exit
               endif
            endif
         enddo
      enddo
      !print*,"wolfprob",wolfprob
      !if ((woff_test)/=wolfprob) then
      !   print*,woff_test,wolfprob
      !   pause
      !endif
      !print*,"clustersize+++++++++++++++++++++++++++++++++++",clustersize
      if (ran()<=wolfprob) then
         !print*,"Yes","weightchange",weightchange
         !print*,"clustersize",sum(updatememory(:,1)),loopnum1,loopnum0,loopnummax
         !print*,"clustersize+++++++++++++++++++++++++++++++++++"
         do i=1,loopnummax
            if (loopstate(i,1)==0d0 .or. updatestate(i,2)==1) then
               cycle
               if (updatememory(i,1)/=0) then
                  print*,"memory"
                  pause
               endif
            else
               loopstate(i,0)=-loopstate(i,0)
               !print*,i,updatestate(i,2),updatememory(i,1)
               !if (i/=lidx_sug) then
               !   print*,"warning"
               !   pause
               !endif
            endif
         enddo
      else
         !print*,"No",weightchange
      endif
      !print*,"clustersize",sum(memory(:,1))
      clustercounter=clustercounter+1
      !print*,clustercounter
      !else
      !clustercounter=clustercounter+1
      !cycle
      !endif
   enddo


end subroutine woffupdate

!==================================================!


!==================================================!

subroutine buildcluster(lidx0)
   use configuration
   implicit none
   integer :: i,lidx0,lidxt,state_nxt,l,j
   real(8) :: extprob,ppp
   real(8), external :: ran

   do l=1,loopnummax
      if (loopop(lidx0,l,0)==0d0 .or. lidx0==l) then
         cycle
      else
      endif
      if (updatememory(l,1)/=0d0) cycle 

      !print*,"check build",lidx0,l,loopstate(lidx0,0),loopstate(l,0),updatestate(lidx0,2),-1
      if (updatememory(l,2)==1) then
         print*,"building error"
         pause
      endif
      !do i=1,12
      !   if (loopop(lidx0,l,i)/=0d0) then
      !      !print*,"check build",i,loopop(lidx0,l,i)
      !   endif
      !enddo
      !print*,"=================================================="
      !print*,lidx0,l,updatestate(lidx0,2),1,updatestate(lidx0,2),-1
      !print*,loopop(lidx0,l,0),loopstate(lidx0,0),loopstate(l,0)
      !print*,loopop(lidx0,l,1:6)
      !print*,loopop(lidx0,l,7:12)
      !print*,lidx0,l
      call opnumchange(lidx0,l,updatestate(lidx0,2),-1,updatestate(lidx0,2),1,1)

      extprob=1d0
      do i=1,3
         do j=0,3
            !print*,i,j,opstatenum(i,j,:),weight(i,j),(opstatenum(i,j,3)-opstatenum(i,j,4))
            if (opstatenum(i,j,4)/=0d0 .or. opstatenum(i,j,3)/=0d0) then
               if (fwight(i,j)/=0d0) then
                  if (opstatenum(i,j,4)/=opstatenum(i,j,3)) then
                     !print*,i,j,opstatenum(i,j,4)-opstatenum(i,j,3),opstatenum(i,j,4),opstatenum(i,j,3)
                     !print*,i,j,opstatenum(i,j,2)-opstatenum(i,j,1),opstatenum(i,j,2),opstatenum(i,j,1)
                     !print*,fwight(i,j),(fwight(i,j)**(opstatenum(i,j,4)-opstatenum(i,j,3)))
                     extprob=extprob*(fwight(i,j)**(opstatenum(i,j,4)-opstatenum(i,j,3)))
                     !print*,extprob
                  endif
               else
                  extprob=0d0
                  pause
                  exit
               endif
            endif
         enddo
      enddo
      woff_test=woff_test*extprob
      extprob=min(extprob,1d0)
      !print*,"extprob",extprob,clustersize
      !print*,"=================================================="

      ppp=1-extprob
      if (ran()<=ppp) then
         updatememory(l,1)=1
         updatememory(l,2)=1
         clustersize=clustersize+1
         updatestate(l,2)=-1
      endif       
   enddo

end subroutine buildcluster

!==================================================!

subroutine opnumchange(l_0,l_1,phase0_old,phase1_old,phase0_new,phase1_new,logp)
   use configuration
   use measurementdata
   implicit none

   integer :: i,j,statebw1,phase0_new,phase1_new,phase0_old,phase1_old
   integer :: k, widx1,widx2,st1,st2,l_0,l_1,logp
 
   opstatenum(:,:,3:4)=0d0
   if (l_1/=l_0) then
      do k=1,12
         st1=int((phase0_old*loopstate(l_0,0)+1)/2)
         st2=int((phase1_old*loopstate(l_1,0)+1)/2)
         !print*,"1.1",st1,st2
         widx1=vex2weight(k,st1,st2,1)
         widx2=vex2weight(k,st1,st2,2)
         !print*,"1.2",widx1,widx2
         opstatenum(widx1,widx2,1+logp*2)=opstatenum(widx1,widx2,1+logp*2)+loopop(l_0,l_1,k)

         st1=int((phase0_new*loopstate(l_0,0)+1)/2)
         st2=int((phase1_new*loopstate(l_1,0)+1)/2)
         !print*,"1.3",st1,st2
         widx1=vex2weight(k,st1,st2,1)
         widx2=vex2weight(k,st1,st2,2)
         !print*,"1.4",widx1,widx2
         opstatenum(widx1,widx2,2+logp*2)=opstatenum(widx1,widx2,2+logp*2)+loopop(l_0,l_1,k)
      enddo
   elseif (l_1==l_0) then
      do k=1,12
         st1=int((phase0_old*loopstate(l_0,0)+1)/2)
         st2=int((phase1_old*loopstate(l_1,0)+1)/2)
         !print*,"1.5",st1,st2
         widx1=vex2weight(k,st1,st2,1)
         widx2=vex2weight(k,st1,st2,2)
         !print*,"1.6",widx1,widx2
         opstatenum(widx1,widx2,1+logp*2)=opstatenum(widx1,widx2,1+logp*2)+loopop(l_0,l_1,k)/2

         st1=int((phase0_new*loopstate(l_0,0)+1)/2)
         st2=int((phase1_new*loopstate(l_1,0)+1)/2)
         !print*,"1.7",st1,st2
         widx1=vex2weight(k,st1,st2,1)
         widx2=vex2weight(k,st1,st2,2)
         !print*,"1.8",widx1,widx2
         opstatenum(widx1,widx2,2+logp*2)=opstatenum(widx1,widx2,2+logp*2)+loopop(l_0,l_1,k)/2
      enddo
   endif

end subroutine opnumchange
!==================================================!


!==================================================!

subroutine preupdate(rank,mm_sug,star,aim)
   use configuration
   use measurementdata
   implicit none
   integer :: mm_sug,rank,update_step,opod_pre,opod_aft,opod_sug,b,update_type,idx_sug,num_sug
   integer :: star,aim
   integer :: j,mm_tmp

   do j=star,aim
      mm_tmp=oporder(j)
      !print*,j,mm_tmp,nh
      if (opstring(mm_tmp)==0) pause
      b=opstring(mm_tmp)/4d0
      call gencurstateop(mm_tmp,b,0)
   enddo

end subroutine preupdate

!==================================================!
subroutine caldloop(mm_pos,b_pos,crsign,linktyp)
   use configuration
   use measurementdata
   implicit none

   integer :: i,k,b_pos,s,mm_pos,vt0,vt1,vt2,vt3,crsign,up_pos,leg_pos,j,si,sortmp,l,sortmp2
   integer :: mmt2,mmt3,tt,lp0,mrk1,loopaft,loopbef,loopmid,mrk2,mrk3,mrk4,vtt,l0,l1,linktyp
   integer, allocatable :: legtag(:,:)
   integer, allocatable :: md(:,:)
   integer, allocatable :: revtab(:)

   allocate(legtag(0:dxl-1,0:2))
   legtag(:,0)=-1
   legtag(:,1)=0
   legtag(:,2)=-1
   headsort(:,0)=-1
   headsort(:,1)=0
   headsort(:,2)=-1
   headsort(:,3)=-1
   headsort(:,4)=-1
   headsort(:,5)=-1
   allocate(revtab(0:dxl-1))
   revtab(:)=-1
   allocate(md(0:3,2))
   md(:,:)=0d0

   midstring(:,:)=0d0
   if (loopnum1/=sum(loopstate(:,1))) then
      print*,"cal0",loopnum1,sum(loopstate(:,1))
      pause
   endif

   !if (crsign==1 .or. crsign==0) then
   !  tt=ltyp_nxt
   !elseif (crsign==-1) then
   !  tt=0
   !endif
   tt=ltyp_nxt
   loopbef=0d0
   loopaft=0d0
   loopmid=0d0
   si=0d0

   !print*,tt,mxl,dxl
   nxtloopidx(:)=0d0   
   sugloopidx(:)=-1d0

   lp0=0d0
   do k=1,mxl
      s=bsites(k,b_pos)
      if (s==-1) cycle
      if (crsign==1) then
         if (custstateop(s)/=-1) then
            vt0=custstateop(s)
            legtag(k-1,1:2)=vertexlist(vt0,1:2)
            midstring(k-1,1:2)=vertexlist(vt0,1:2)
            vt1=vertexlist_map(vt0)
            !print*,"------add------",int(vt0/dxl),k-1,vertexlist(vt0,1:2),int(vt1/dxl),k-1+mxl,vertexlist(vt1,1:2)
            legtag(k-1+mxl,1:2)=vertexlist(vt1,1:2)
            midstring(k-1+mxl,1:2)=legtag(k-1+mxl,1:2)
         else
            lp0=lp0+1
            legtag(k-1,1)=-lp0
            legtag(k-1+mxl,1)=-lp0
            legtag(k-1,2)=0
            legtag(k-1+mxl,2)=s
            !print*,"insert",s,b_pos,k,ltyp_nxt
            midstring(k-1,1:2)=legtag(k-1,1:2)
            midstring(k-1+mxl,1:2)=legtag(k-1+mxl,1:2)
         endif
      elseif (crsign==-1 .or. crsign==0d0) then
         vt0=dxl*mm_pos+k-1
         legtag(k-1,1:2)=vertexlist(vt0,1:2)
         midstring(k-1,1:2)=legtag(k-1,1:2)
         vt1=dxl*mm_pos+k-1+mxl
         !print*,"------rem------",int(vt0/dxl),k-1,vertexlist(vt0,1:2),int(vt1/dxl),k-1+mxl,vertexlist(vt1,1:2)
         legtag(k-1+mxl,1:2)=vertexlist(vt1,1:2)
         midstring(k-1+mxl,1:2)=legtag(k-1+mxl,1:2)
      endif
   enddo
   !read out the loop structure

   !print*,"check"

   sortmp=0d0
   sortmp2=0d0
   do i=0,dxl-1
      si=bsites(mod(i,mxl)+1,b_pos)
      if (si==-1) cycle
      if (legtag(i,0)==-1) then
         mrk1=legtag(i,1)
         headsort(sortmp,0)=i
         headsort(sortmp,1)=legtag(i,1)
         headsort(sortmp,2)=legtag(i,2)
         headsort(sortmp,3)=sortmp
         revtab(i)=sortmp
         legtag(i,0)=legtag(i,0)+1
         loopbef=loopbef+1

         do j=0,dxl-1
            si=bsites(mod(j,mxl)+1,b_pos)
            if (si==-1) cycle
            mrk2=legtag(j,1)
            if (legtag(j,0)==-1 .and. mrk1==mrk2) then
               l=j
               md(0,1)=l
               md(1,1)=mrk2
               md(2,1)=legtag(l,2)
               md(3,1)=sortmp

               do k=sortmp,dxl-1
                  if (headsort(k,1)/=mrk2 .and. headsort(k,1)/=0) then
                     print*,"sort error",headsort(k,1),mrk2 
                     pause
                  endif
                  if (headsort(k,2)==-1) then
                     headsort(k,:)=md(:,1)
                     revtab(l)=k
                     sortmp2=k
                     exit
                  elseif (headsort(k,2)<legtag(j,2)) then
                  elseif (headsort(k,2)>legtag(j,2)) then
                     md(:,2)=headsort(k,:)
                     headsort(k,:)=md(:,1)
                     md(:,1)=md(:,2)
                     revtab(l)=k
                     l=md(0,2)
                  endif
               enddo
               legtag(j,0)=legtag(j,0)+1
            endif
         enddo
         sortmp=sortmp2+1
         !print*,sortmp,sortmp2
      endif
   enddo
   !sort the leg-index
   !print*,"check"

   do i=0,dxl-1
      if (headsort(i,2)==-1) exit
      l0=headsort(i,0)
      l1=headsort(mod(i+1,dxl),0)
      !print*,"i",i
      if (i==headsort(i,3) .and. &
         &(headsort(headsort(i,3),2)/=1 .or. crsign==-1 .or. crsign==0d0)) then
         !head
         loopmid=loopmid+1
         headsort(i,4)=loopmid
         headsort(i,5)=loopmid
         midstring(headsort(i,0),3)=2*loopmid+1
         legtag(l0,0)=legtag(l0,0)+1
      elseif ((headsort(i,3)/=headsort(mod(i+1,dxl),3) .or. mod(i+1,dxl)==headsort(i,3)) .and. &
         &(headsort(headsort(i,3),2)/=1 .or. crsign==-1 .or. crsign==0d0)) then
         !tail
         headsort(i,4)=headsort(headsort(i,3),4)
         headsort(i,5)=headsort(headsort(i,3),5)
         midstring(headsort(i,0),3)=2*headsort(headsort(i,3),4)+1
         legtag(l0,0)=legtag(l0,0)+1
         midstring(headsort(i,0),4)=headsort(headsort(i,3),0)
         midstring(headsort(headsort(i,3),0),4)=headsort(i,0)
      elseif (headsort(i,3)==headsort(mod(i+1,dxl),3)) then
         !print*,"l0,l1",l0,l1,loopmid
         !print*,"con",legtag(:,0)
         !print*,headsort(:,0)
         !print*,"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"
         if (legtag(l0,0)==0) then
            vt0=i-headsort(i,3)
            vt1=i+1-headsort(mod(i+1,dxl),3)
            if (vt1==vt0+1 .and. headsort(i,3)==headsort(mod(i+1,dxl),3)) then
               loopmid=loopmid+1
               headsort(i,4)=loopmid
               headsort(i,5)=loopmid
               midstring(headsort(i,0),3)=2*loopmid
               headsort(mod(i+1,dxl),4)=loopmid
               headsort(mod(i+1,dxl),5)=loopmid
               midstring(headsort(mod(i+1,dxl),0),3)=2*loopmid
               legtag(l1,0)=legtag(l1,0)+1
               midstring(headsort(mod(i+1,dxl),0),4)=headsort(i,0)
               midstring(headsort(i,0),4)=headsort(mod(i+1,dxl),0)
               !print*,"add l1",l1,legtag(l1,0)
            else
            endif
            legtag(l0,0)=legtag(l0,0)+1
            !print*,"add l0",l0,legtag(l0,0)
         elseif (legtag(l0,0)==1) then
         endif
      endif
   enddo

   !print*,"---------------------------------------00000000000000"
   !print*,"l0",legtag(:,0)
   !print*,"l1",legtag(:,1)
   !print*,"l2",legtag(:,2)
   !print*,"-----------------------------------------------------"
   !print*,"0",headsort(:,0)!leg_idx
   !print*,"1",headsort(:,1)!org_loop_idx
   !print*,"2",headsort(:,2)!org_loop_ord
   !print*,"3",headsort(:,3)!sort_helper
   !print*,"4",headsort(:,4)!string_idx
   !print*,"rev",revtab(:)
   !print*,"loopbef",loopbef
   !print*,"check"   

   do i=0,dxl-1
      !print*,i
      si=bsites(mod(i,mxl)+1,b_pos)
      if (si==-1) cycle
      !print*,i,legtag(revtab(i),0),headsort(revtab(i),4),&
      !&legtag(i,0)==1 .and. (headsort(revtab(i),4)/=0 .and. headsort(revtab(i),4)/=-1)
      if (legtag(i,0)==1 .and. (headsort(revtab(i),4)/=0 .and. headsort(revtab(i),4)/=-1)) then

         !print*,i,tt,linktable(l0,tt),"test"
         l0=i
         l1=linktable(l0,tt)
         !print*,vt0,vt1

         vt0=headsort(revtab(l0),4)
         vt1=headsort(revtab(l1),4)
         !print*,vt0,vt1
         if (vt0/=vt1) then
            do j=0,dxl-1
               if (headsort(j,4)==vt1) then
                  if (j==revtab(l1)) then
                     headsort(j,4)=0d0
                     midstring(headsort(j,0),5)=loopaft+1
                  else
                     headsort(j,4)=vt0
                  endif
               endif
               if (headsort(j,4)==vt0) then
                  if (j==revtab(l0)) then
                     headsort(j,4)=0d0
                     midstring(headsort(j,0),5)=loopaft+1
                  else
                     headsort(j,4)=vt0
                  endif
               endif
            enddo
         elseif (vt0==vt1) then
            do j=0,dxl-1
               if (headsort(j,4)==vt0) then
                  headsort(j,4)=0
                  midstring(headsort(j,0),5)=loopaft+1
               endif
            enddo    
            loopaft=loopaft+1    
         endif
         !print*,"l0",legtag(:,0)
         !print*,"4",headsort(:,4)
         !print*,"4444444444444444444444444444444",loopaft
      endif
   enddo
   dloop=loopaft-loopbef
   midstring(0,0)=loopbef
   midstring(1,0)=loopmid
   midstring(2,0)=loopaft
   midstring(3,0)=dloop

   jdgstring(:,:)=-1d0
   do i=0,dxl-1
      if (jdgstring(int(midstring(i,3)/2),0)==-1) then
         jdgstring(int(midstring(i,3)/2),0)=midstring(i,3)
         jdgstring(int(midstring(i,3)/2),1)=midstring(i,1)
         jdgstring(int(midstring(i,3)/2),2)=midstring(i,5)
         jdgstring(int(midstring(i,3)/2),3)=min(midstring(i,2),midstring(midstring(i,4),2))
         jdgstring(int(midstring(i,3)/2),4)=max(midstring(i,2),midstring(midstring(i,4),2))
         jdgstring(int(midstring(i,3)/2),5)=min(i,midstring(i,4))
         jdgstring(int(midstring(i,3)/2),6)=max(i,midstring(i,4))
      endif
   enddo
   jdgstring(:,7:8)=0d0
   !print*,"================================================"
   !print*,ltyp_org,ltyp_nxt,mm_pos
   !print*,nxtloopidx(:)
   !print*,"st0",midstring(:,0)!number,org_loop,string,nxt_loop
   !print*,"st1",midstring(:,1)!loop connect of org
   !print*,"st2",midstring(:,2)!loop order of org
   !print*,"st3",midstring(:,3)!loop connect of string
   !print*,"st4",midstring(:,4)!loop connecting of string
   !print*,"st5",midstring(:,5)!loop connect of next
   !print*,"------------------------------------------------"

   !print*,"jg0",jdgstring(:,0)!loop connect of string
   !print*,"jg1",jdgstring(:,1)!loop connect of org
   !print*,"jg2",jdgstring(:,2)!loop connect of next
   !print*,"jg3",jdgstring(:,3)
   !print*,"jg4",jdgstring(:,4)
   !print*,"jg5",jdgstring(:,5)
   !print*,"jg6",jdgstring(:,6)
   !print*,"chgloopidx",chgloopidx(:,0)
   !print*,"chgloopidx",chgloopidx(:,1)
   !print*,"================================================"
   !print*,"sot0",headsort(:,0)!leg_idx
   !print*,"sot1",headsort(:,1)!org_loop_idx
   !print*,"sot2",headsort(:,2)!org_loop_ord
   !print*,"sot3",headsort(:,3)!sort_helper
   !print*,"sot4",headsort(:,4)!string_idx
   !print*,"sot5",headsort(:,5)!string_idx

   !pause
   !print*,"initstringrelationtab"
   call initstringrelationtab(mm_pos,b_pos,crsign)

   !print*,"check0"

   !print*,revtab(:)
   !print*,"loopaft,loopbef,loopmid,dloop,crsign,tt"
   !print*,loopaft,loopbef,loopmid,dloop,crsign,tt
   !print*,"0",headsort(:,0)
   !print*,"1",headsort(:,1)
   !print*,"2",headsort(:,2)
   !print*,"3",headsort(:,3)
   !print*,"4",headsort(:,4)
   !print*,"5",headsort(:,5)
   !print*,"+++++++++++++++++++++++++++++++"
   !print*,"st0",midstring(:,0)!number,org_loop,string,nxt_loop
   !print*,"st1",midstring(:,1)!loop connect of org
   !print*,"st2",midstring(:,2)!loop order of org
   !print*,"st3",midstring(:,3)!loop connect of string
   !print*,"st4",midstring(:,4)!loop connecting of string
   !print*,"st5",midstring(:,5)!loop connect of next
   !do i=0,dxl-1
   !   if (midstring(midstring(i,4),3)/=midstring(i,3)) then
   !      pause
   !   endif
   !enddo
   !if (dloop/=1 .and. dloop/=0 .and. dloop/=-1) then
   !  print*,"limit",dloop
   !endif

end subroutine caldloop

!==================================================!
!==================================================!
subroutine initstringrelationtab(mm_pos,b_pos,crsign)
   use configuration
   use vertexupdate
   integer :: i,j,k,l,stridx0,stridx1,stridx2,stridx3,vt,b_pos,s,mm_pos,crsign
   integer :: mm_tmp,lidx_rel,lord_tmp,lidx_tmp,leg_tmp,ltyp_tmp,leg_tmp2,phase_tmp
   integer :: vtt,lidx_tmp2,lord_tmp2,str_idx0,str_idx2,step_cur,step_len,step,step_par
   integer :: step_sta2,step_end2,loadtyp,loadtyp2,dphase_nxt,dphase_org,dphase
   integer :: str_idx,str_jdg,op_typ,vmin,vmax,strsign0,strsign2,step_sta,step_end
   integer :: step_cur2,step_len2,step2,step_par2,l0,l1,l2,st1,st2
   integer :: widx1,widx2,vex2record_nxt,vex2record_org,widx1_nxt,widx2_nxt,widx1_org,widx2_org
   real(8) :: weight_org
   integer :: lt1,lt2,lt3,lt4
   integer, allocatable :: check(:)
   integer, allocatable :: visitab(:)
   integer, allocatable :: strtab(:,:)
   integer, allocatable :: strstrel(:,:,:)
   integer, allocatable :: dstrel(:,:,:)
   integer, allocatable :: phasechang(:)
   integer, allocatable :: weight_count(:,:)
   real(8), external :: ran
   !print*,"init string",ltyp_org,ltyp_nxt

   allocate(visitab(0:dxl-1))
   visitab(:)=0d0
   allocate(strtab(dxl,0:7))
   strtab(:,:)=0d0
   allocate(strstrel(dxl,dxl,0:12))
   strstrel(:,:,:)=0d0
   allocate(dstrel(dxl,dxl,0:12))
   dstrel(:,:,:)=0d0
   allocate(check(0:12))
   check(:)=0d0
   nextst(:,:,:)=0d0
   opstatenum(:,:,:)=0d0
   allocate(phasechang(1:dxl))
   phasechang(:)=1d0

   lidx_rel=headsort(0,1)
   j=1
   if (lidx_rel>0) then
      loopstate(headsort(0,1),2)=j
      strtab(j,0)=lidx_rel
      strtab(j,1)=0
      strtab(j,3)=headsort(0,5)
      strtab(j,4)=headsort(0,5)
   else
      strtab(j,0)=lidx_rel  
      strtab(j,1)=0 
      strtab(j,3)=headsort(0,5)
      strtab(j,4)=headsort(0,5)
   endif
   do i=1,dxl-1
      if (i==dxl-1) then
         strtab(j,2)=i
         strtab(j,4)=max(strtab(j,4),headsort(i,5))
      elseif (lidx_rel/=headsort(i,1)) then
         if (headsort(i,1)>0) then
            strtab(j,2)=i-1
            j=j+1 
            loopstate(headsort(i,1),2)=j
            lidx_rel=headsort(i,1)
            strtab(j,0)=lidx_rel
            strtab(j,1)=i
            strtab(j,3)=headsort(i,5)
            strtab(j,4)=headsort(i,5)
         else
            strtab(j,2)=i-1
            j=j+1 
            lidx_rel=headsort(i,1)
            strtab(j,0)=lidx_rel  
            strtab(j,1)=i
            strtab(j,3)=headsort(i,5)
            strtab(j,4)=headsort(i,5)
         endif
      else
         strtab(j,4)=max(strtab(j,4),headsort(i,5))
      endif
   enddo
   !print*,"vis",visitab(:)
   !print*,"str0",strtab(:,0)
   !print*,"str1",strtab(:,1)
   !print*,"str2",strtab(:,2)
   !print*,"str3",strtab(:,3)
   !print*,"str4",strtab(:,4)
   dprob=1d0
   !sprob=1d0
   midlstate(:,:)=0d0
   nxtphase(:)=0d0
   do i=0,mxl-1
      do j=0,1
         l0=i+j*mxl
         l2=linktable(l0,ltyp_nxt)
         !print*,"gen nxtp",l0,l2,nxtphase(l0),nxtphase(l2)
         if (nxtphase(l0)==0d0 .and. nxtphase(l2)==0d0) then
            if (crsign==1 .or. crsign==0) then
               nxtphase(l0)=1-2*j
            elseif (crsign==-1) then
               nxtphase(l0)=-(1-2*j)
            endif
            str_idx=int(midstring(l0,3)/2d0)
            !print*,str_idx
            jdgstring(str_idx,7)=nxtphase(l0)*curphase(l0)
            jdgstring(str_idx,8)=nxtphase(l0)
            !strtab(str_idx,5)=nxtphase(l0)*curphase(l0)
            !strtab(str_idx,6)=curphase(l0)
            !strtab(str_idx,7)=nxtphase(l0)
            l1=jdgstring(str_idx,5)
            !l1=headsort(strtab(str_idx,1),0)
            if (nxtphase(l1)==0d0) then
               !nxtphase(l1)=strtab(str_idx,5)*curphase(l1)
               nxtphase(l1)=jdgstring(str_idx,7)*curphase(l1)
            endif
            !print*,"l1_1",l1,nxtphase(l1)
            l1=jdgstring(str_idx,6)
            !l1=headsort(strtab(str_idx,2),0)
            if (nxtphase(l1)==0d0) then
               !nxtphase(l1)=strtab(str_idx,5)*curphase(l1)
               nxtphase(l1)=jdgstring(str_idx,7)*curphase(l1)
            endif
            !print*,"l1_1",l1,nxtphase(l1)
            !if (nxtphase(l2)==0d0) then
            !   if (ltyp_nxt==1) then
            !      nxtphase(l2)=-nxtphase(l0)
            !   elseif (ltyp_nxt==0 .or. ltyp_nxt==2) then
            !      nxtphase(l2)=nxtphase(l0)
            !   endif
            !endif
            !print*,"l2_1",l2,nxtphase(l2)
         elseif (nxtphase(l0)==0d0 .and. nxtphase(l2)/=0d0) then
            if (ltyp_nxt==1) then
               nxtphase(l0)=-nxtphase(l2)
            elseif (ltyp_nxt==0 .or. ltyp_nxt==2) then
               nxtphase(l0)=nxtphase(l2)
            endif
            !print*,"l0_2",l0,nxtphase(l0),nxtphase(l2)
            str_idx=int(midstring(l0,3)/2d0)
            !print*,str_idx
            jdgstring(str_idx,7)=nxtphase(l0)*curphase(l0)
            jdgstring(str_idx,8)=nxtphase(l0)
            !strtab(str_idx,5)=nxtphase(l0)*curphase(l0)
            !strtab(str_idx,6)=curphase(l0)
            !strtab(str_idx,7)=nxtphase(l0)
            l1=jdgstring(str_idx,5)
            !l1=headsort(strtab(str_idx,1),0)
            if (nxtphase(l1)==0d0) then
               !nxtphase(l1)=strtab(str_idx,5)*curphase(l1)
               nxtphase(l1)=jdgstring(str_idx,7)*curphase(l1)
            endif
            !print*,"l1_2",l1,nxtphase(l1)
            l1=jdgstring(str_idx,6)
            !l1=headsort(strtab(str_idx,2),0)
            if (nxtphase(l1)==0d0) then
               !nxtphase(l1)=strtab(str_idx,5)*curphase(l1)
               nxtphase(l1)=jdgstring(str_idx,7)*curphase(l1)
            endif
            !print*,"l1_2",l1,nxtphase(l1)
         elseif (nxtphase(l0)/=0d0) then
            cycle 
         endif
      enddo
   enddo

   do i=0, dxl-1
      l0=midstring(i,5)
      !print*,"i",i,l0,midlstate(:,0)
      if (midlstate(l0,0)==0d0) then
         midlstate(l0,0)=curstate(i)*curphase(i)*nxtphase(i)
         !print*,curstate(i),curphase(i),nxtphase(i)
      endif
   enddo

   do i=0,dxl-1
      l0=midstring(i,5)
      if (midlstate(l0,0)*nxtphase(i)/=curstate(i)*curphase(i)) then
         print*,"++++++++++++++++++++++++++++++++++++++++++++++++",ltyp_nxt,ltyp_org,crsign
         do j=0,dxl-1
            l0=midstring(j,5)
            print*,"leg",j,midstring(j,5),midlstate(l0,0),curstate(j),nxtphase(j),curphase(j)
         enddo
         print*,"state error"
         print*,"++++++++++++++++++++++++++++++++++++++++++++++++"
         pause
      endif
   enddo

   !print*,"ltyp_org",ltyp_org,"ltyp_nxt",ltyp_nxt
   !print*,"++++++++++++++++++++++++++++++++++++++++++++++++"
   !print*,"st1",midstring(:,1)!loop connect of org
   !print*,"cs",curstate(:)
   !print*,"cp",curphase(:)
   !print*,"++++++++++++++++++++++++++++++++++++++++++++++++"
   !print*,"st",int(midstring(:,3)/2)!loop connect of string
   !print*,jdgstring(:,7)
   !print*,jdgstring(:,8)
   !print*,"++++++++++++++++++++++++++++++++++++++++++++++++"
   !print*,"st5",midstring(:,5)!loop connect of next
   !print*,"ms",midlstate(:,0)
   !print*,"np",nxtphase(:)
   !print*,"++++++++++++++++++++++++++++++++++++++++++++++++"

   if (dprob==1) then
      if (midstring(1,0)/=2) then
         print*,"string number"
         pause
      else
         do i=1,dxl
            do j=1,loopnummax
               if (nextlp(i,j,0)/=0d0) then
                  nextlp(i,j,:)=0d0
               endif
            enddo
            if (jdgstring(i,0)/=-1) then
               do j=1,dxl
                  if (i/=j .and. jdgstring(j,0)/=-1) then
                     if (crsign==1) then
                        phase2jdg(:,1)=midstring(:,5)
                        phase2jdg(:,2)=nxtphase(:)
                        !print*,"nxt",phase2jdg(:,1)
                        !print*,"nxt",phase2jdg(:,2)
                        call phase2vex(ltyp_nxt,jdgstring(i,2),jdgstring(j,2))

                        dstrel(i,j,0)=dstrel(i,j,0)+1
                        dstrel(i,j,vex2record)=dstrel(i,j,vex2record)+1

                        nextst(jdgstring(i,2),jdgstring(j,2),0)=&
                        &nextst(jdgstring(i,2),jdgstring(j,2),0)+1
                        nextst(jdgstring(i,2),jdgstring(j,2),vex2record)=&
                        &nextst(jdgstring(i,2),jdgstring(j,2),vex2record)+1

                        if (jdgstring(i,2)>0d0 .and. jdgstring(j,2)>0d0) then
                           st1=int((midlstate(jdgstring(i,2),0)+1)/2)
                           st2=int((midlstate(jdgstring(j,2),0)+1)/2)
                           !print*,vex2record,st1,st2
                           widx1=vex2weight(vex2record,st1,st2,1)
                           widx2=vex2weight(vex2record,st1,st2,2)
                           opstatenum(0,widx2,2)=&
                           &opstatenum(0,widx2,2)+1d0
                           opstatenum(widx1,widx2,2)=&
                           &opstatenum(widx1,widx2,2)+1d0
                           !print*,"nxt 1",widx1,widx2,ltyp_nxt,weight(widx1,widx2)
                           if (weight(widx1,widx2)==0d0) then
                           !   print*,widx1,widx2
                           !   print*,ltyp_nxt,statebw,midlstate(jdgstring(i,2),0),midlstate(jdgstring(j,2),0)
                           !   print*,jdgstring(i,2),jdgstring(j,2)
                              print*,"error nxt 1"
                              pause
                           endif
                        endif
                        !print*,i,j,jdgstring(i,2),jdgstring(j,2)
                     elseif (crsign==-1) then

                        phase2jdg(:,1)=midstring(:,1)
                        phase2jdg(:,2)=curphase(:)
                        !print*,"org",phase2jdg(:,1)
                        !print*,"org",phase2jdg(:,2)
                        call phase2vex(ltyp_org,jdgstring(i,1),jdgstring(j,1))

                        dstrel(i,j,0)=dstrel(i,j,0)-1
                        dstrel(i,j,vex2record)=dstrel(i,j,vex2record)-1

                        nextst(jdgstring(i,2),jdgstring(j,2),0)=&
                        &nextst(jdgstring(i,2),jdgstring(j,2),0)-1
                        nextst(jdgstring(i,2),jdgstring(j,2),vex2record)=&
                        &nextst(jdgstring(i,2),jdgstring(j,2),vex2record)-1

                        if (jdgstring(i,1)>0d0 .and. jdgstring(j,1)>0d0) then
                           st1=int((loopstate(jdgstring(i,1),0)+1)/2)
                           st2=int((loopstate(jdgstring(j,1),0)+1)/2) 
                           widx1=vex2weight(vex2record,st1,st2,1)
                           widx2=vex2weight(vex2record,st1,st2,2)
                           opstatenum(0,widx2,1)=&
                           &opstatenum(0,widx2,1)+1d0
                           opstatenum(widx1,widx2,1)=&
                           &opstatenum(widx1,widx2,1)+1d0
                           !print*,"org 1",widx1,widx2,ltyp_org,vex2record
                           if (weight(widx1,widx2)==0d0) then
                              print*,ltyp_org,dphase_org,statebw,loopstate(jdgstring(i,1),0),loopstate(jdgstring(j,1),0)
                              print*,jdgstring(i,1),jdgstring(j,1)
                              print*,"error org 1"
                              pause
                           endif
                        endif

                        !print*,i,j,jdgstring(i,2),jdgstring(j,2)
                     elseif (crsign==0) then
                        phase2jdg(:,1)=midstring(:,5)
                        phase2jdg(:,2)=nxtphase(:)
                        !print*,"nxt",phase2jdg(:,1)
                        !print*,"nxt",phase2jdg(:,2)
                        call phase2vex(ltyp_nxt,jdgstring(i,2),jdgstring(j,2))
                        vex2record_nxt=vex2record

                        phase2jdg(:,1)=midstring(:,1)
                        phase2jdg(:,2)=curphase(:)
                        !print*,"org",phase2jdg(:,1)
                        !print*,"org",phase2jdg(:,2)
                        call phase2vex(ltyp_org,jdgstring(i,1),jdgstring(j,1))
                        vex2record_org=vex2record

                        dstrel(i,j,0)=dstrel(i,j,0)
                        dstrel(i,j,vex2record_nxt)=dstrel(i,j,vex2record_nxt)+1
                        dstrel(i,j,vex2record_org)=dstrel(i,j,vex2record_org)-1

                        nextst(jdgstring(i,2),jdgstring(j,2),0)=&
                        &nextst(jdgstring(i,2),jdgstring(j,2),0)+1
                        nextst(jdgstring(i,2),jdgstring(j,2),vex2record_nxt)=&
                        &nextst(jdgstring(i,2),jdgstring(j,2),vex2record_nxt)+1

                        nextst(jdgstring(i,2),jdgstring(j,2),0)=&
                        &nextst(jdgstring(i,2),jdgstring(j,2),0)-1
                        nextst(jdgstring(i,2),jdgstring(j,2),vex2record_org)=&
                        &nextst(jdgstring(i,2),jdgstring(j,2),vex2record_org)-1

                        if (jdgstring(i,2)>0d0 .and. jdgstring(j,2)>0d0) then
                           st1=int((midlstate(jdgstring(i,2),0)+1)/2)
                           st2=int((midlstate(jdgstring(j,2),0)+1)/2)
                           !print*,vex2record_nxt,st1,st2
                           widx1=vex2weight(vex2record_nxt,st1,st2,1)
                           widx2=vex2weight(vex2record_nxt,st1,st2,2)
                           opstatenum(0,widx2,2)=&
                           &opstatenum(0,widx2,2)+1d0
                           opstatenum(widx1,widx2,2)=&
                           &opstatenum(widx1,widx2,2)+1d0
                           !print*,"nxt 2",widx1,widx2,ltyp_nxt
                           if (weight(ltyp_nxt+1,statebw)==0d0) then
                              print*,ltyp_nxt,statebw,midlstate(jdgstring(i,2),0),midlstate(jdgstring(j,2),0)
                              print*,jdgstring(i,2),jdgstring(j,2)
                              print*,"error nxt 2"
                              pause
                           endif
                        endif

                        if (jdgstring(i,1)>0d0 .and. jdgstring(j,1)>0d0) then
                           st1=int((loopstate(jdgstring(i,1),0)+1)/2)
                           st2=int((loopstate(jdgstring(j,1),0)+1)/2)
                           !print*,vex2record_org,st1,st2
                           widx1=vex2weight(vex2record_org,st1,st2,1)
                           widx2=vex2weight(vex2record_org,st1,st2,2)
                           opstatenum(0,widx2,1)=&
                           &opstatenum(0,widx2,1)+1d0
                           opstatenum(widx1,widx2,1)=&
                           &opstatenum(widx1,widx2,1)+1d0
                           !print*,"org 2",widx1,widx2,ltyp_org
                           if (weight(widx1,widx2)==0d0) then
                              print*,ltyp_org,statebw,loopstate(jdgstring(i,1),0),loopstate(jdgstring(j,1),0)
                              print*,jdgstring(i,1),jdgstring(j,1)
                              print*,"error org 2"
                              pause
                           endif
                        endif
                        !print*,i,j,jdgstring(i,2),jdgstring(j,2)
                     endif
                  endif
               enddo
            endif
         enddo
      endif
      if (sum(opstatenum(0,:,2))/2-sum(opstatenum(0,:,1))/2/=crsign) then
         print*,"sum err 0",sum(opstatenum(0,:,2)),sum(opstatenum(0,:,1)),crsign
         pause
      endif
      !print*,"d op"
      !print*,"check"

      do i=1,dxl
         if (strtab(i,0)==0) then
            exit
         else
            lidx_rel=strtab(i,0)
            if ( lidx_rel>0 ) then
               !print*,"relating loop basis",lidx_rel
               do j=1, loopnummax
                  !print*,"clean",jdgstring(i,2),j
                  if (loopoprecord(lidx_rel,j,0)==0d0) then
                     cycle
                  else
                     strgop(strtab(i,3):strtab(i,4),j,:)=0d0
                     step_sta=strtab(loopstate(lidx_rel,2),1)
                     step_end=strtab(loopstate(lidx_rel,2),2)
                     step_len=(step_end-step_sta)+1
                     step_cur=step_sta
                     !dphase=strtab(i,5)
                     do l=1,loopoprecord(lidx_rel,j,0)
                        vt=loopoprecord(lidx_rel,j,l)
                        mm_tmp=int(vt/dxl)
                        lidx_tmp=vertexlist(vt,1)
                        lord_tmp=vertexlist(vt,2)
                        op_typ=mod(opstring(int(vt/dxl)),4)
                        phase2jdg(:,1)=vertexlist(mm_tmp*dxl:mm_tmp*dxl+dxl-1,1)
                        lt1=mod(vt,dxl)
                        lt2=linktable(lt1,op_typ)
                        loadtyp=0d0
                        do 
                           step_par=mod(step_cur-step_sta+1,step_len)+step_sta
                           if (headsort(step_par,5)/=headsort(step_cur,5)) then
                              step_par=step_end
                              step=1
                              loadtyp=1d0
                              if (headsort(step_par,5)/=headsort(step_cur,5)) then
                              endif
                           else
                              step=2
                           endif
                           str_idx=headsort(step_cur,5)
                           str_jdg=mod(jdgstring(str_idx,0),2)
                           vmin=jdgstring(str_idx,3)
                           vmax=jdgstring(str_idx,4)
                           !dphase=strtab(str_idx,5)
                           dphase=jdgstring(str_idx,7)
                           phase2jdg(lt1,1)=-str_idx
                           phase2jdg(lt2,1)=-str_idx
                           if (str_jdg==0) then
                              if (lord_tmp>=vmin .and. lord_tmp<=vmax) then
                                 exit
                              endif
                           elseif (str_jdg==1) then
                              if (lord_tmp<=vmin .or. lord_tmp>=vmax) then
                                 exit
                              endif
                           endif
                           !print*,step_cur,step,step_len
                           if (step_cur==mod(step_cur+step-step_sta,step_len)+step_sta) then
                              print*,"warning",lidx_rel,lord_tmp
                              print*,"warning",str_idx,str_jdg,vmin,vmax
                              pause
                           endif
                           step_cur=mod(step_cur+step-step_sta,step_len)+step_sta
                           !print*,step_cur,step,step_len
                        enddo

                        if (loopstate(j,2)==0d0) then
                           !print*,"str_idx",str_idx,"lidx_rel",lidx_rel,lord_tmp,j
                           !print*,"judge",str_jdg,int(vt/dxl),vmin,vmax
                           !print*,"-2.0-",int(vt/dxl)*dxl,int(vt/dxl)*dxl+1,int(vt/dxl)*dxl+2,int(vt/dxl)*dxl+3
                           !print*,"-2.1-",vertexlist(int(vt/dxl)*dxl:int(vt/dxl)*dxl+dxl-1,1)
                           !print*,"-2.2-",vertexlist(int(vt/dxl)*dxl:int(vt/dxl)*dxl+dxl-1,2)
                           !print*,"strop1",strgop(str_idx,j,:)

                           phase2jdg(:,2)=vertexlist(mm_tmp*dxl:mm_tmp*dxl+dxl-1,5)
                           !print*,"org",phase2jdg(:,1)
                           !print*,"org",phase2jdg(:,2)
                           call phase2vex(op_typ,-str_idx,j)
                           vex2record_org=vex2record
                           do k=0,dxl-1
                              if (phase2jdg(k,1)==-str_idx) then
                                 phase2jdg(k,2)=phase2jdg(k,2)*dphase
                              endif
                           enddo
                           !print*,"nxt",phase2jdg(:,1)
                           !print*,"nxt",phase2jdg(:,2)
                           call phase2vex(op_typ,-str_idx,j)
                           vex2record_nxt=vex2record

                           strgop(str_idx,j,0)=strgop(str_idx,j,0)+1
                           strgop(str_idx,j,vex2record)=strgop(str_idx,j,vex2record_nxt)+1
                           !print*,"strop2",strgop(str_idx,j,:)

                           nextlp(jdgstring(str_idx,2),j,0)=&
                           &nextlp(jdgstring(str_idx,2),j,0)+1
                           nextlp(jdgstring(str_idx,2),j,vex2record_nxt)=&
                           &nextlp(jdgstring(str_idx,2),j,vex2record_nxt)+1

                           if (jdgstring(str_idx,2)>0d0 .and. j>0d0) then
                              st1=int((midlstate(jdgstring(str_idx,2),0)+1)/2)
                              st2=int((loopstate(j,0)+1)/2)
                              !print*,"nxt 3",vex2record_nxt,st1,st2
                              widx1_nxt=vex2weight(vex2record_nxt,st1,st2,1)
                              widx2_nxt=vex2weight(vex2record_nxt,st1,st2,2)
                              !print*,"nxt 3",widx1_nxt,widx2_nxt
                              opstatenum(0,widx2_nxt,2)=&
                              &opstatenum(0,widx2_nxt,2)+2d0
                              opstatenum(widx1_nxt,widx2_nxt,2)=&
                              &opstatenum(widx1_nxt,widx2_nxt,2)+2d0
                              !print*,"nxt 3","================================"
                              if (weight(widx1_nxt,widx2_nxt)==0d0) then
                                 !print*,op_typ,statebw,midlstate(jdgstring(str_idx,2),0),loopstate(j,0)
                                 !print*,jdgstring(str_idx,2),j
                                 !print*,"error nxt 3"
                                 !pause
                              endif
                           endif

                           if (jdgstring(str_idx,1)>0d0 .and. j>0d0) then
                              st1=int((loopstate(jdgstring(str_idx,1),0)+1)/2)
                              st2=int((loopstate(j,0)+1)/2)
                              !print*,"org 3",vex2record_org,st1,st2
                              widx1_org=vex2weight(vex2record_org,st1,st2,1)
                              widx2_org=vex2weight(vex2record_org,st1,st2,2)
                              !print*,"org 3",widx1_org,widx2_org
                              opstatenum(0,widx2_org,1)=&
                              &opstatenum(0,widx2_org,1)+2d0
                              opstatenum(widx1_org,widx2_org,1)=&
                              &opstatenum(widx1_org,widx2_org,1)+2d0
                              !print*,"org 3","================================"
                              if (weight(widx1_org,widx2_org)==0d0) then
                                 !print*,op_typ,statebw,loopstate(jdgstring(str_idx,1),0),loopstate(j,0)
                                 !print*,jdgstring(str_idx,1),j
                                 !print*,"error org 3"
                                 !pause
                              endif
                           endif

                           if (widx1_nxt/=widx1_org .or. widx2_nxt/=widx2_org) then
                              print*,"nxt err1",widx1_nxt,widx2_nxt
                              print*,"org err1",widx1_org,widx2_org
                              pause
                           endif
                        elseif (loopstate(j,2)/=0d0) then
!=========================================================================================!
                           step_sta2=strtab(loopstate(j,2),1)
                           step_end2=strtab(loopstate(j,2),2)
                           step_len2=(step_end2-step_sta2)+1
                           step_cur2=step_sta2
                           vtt=vertexlist(vt,4)
                           lidx_tmp2=vertexlist(vtt,1)
                           lord_tmp2=vertexlist(vtt,2)  
                           op_typ=mod(opstring(mm_tmp),4)
                           lt3=mod(vtt,dxl)
                           lt4=linktable(lt3,op_typ)
                           loadtyp2=0d0
                           do 
                              step_par2=mod(step_cur2-step_sta2+1,step_len2)+step_sta2
                              if (headsort(step_par2,5)/=headsort(step_cur2,5)) then
                                 step_par2=step_end2
                                 step2=1
                                 loadtyp2=1d0
                              else
                                 step2=2
                              endif
                              str_idx2=headsort(step_cur2,5)
                              str_jdg2=mod(jdgstring(str_idx2,0),2)
                              vmin=jdgstring(str_idx2,3)
                              vmax=jdgstring(str_idx2,4)
                              phase2jdg(lt3,1)=-str_idx2
                              phase2jdg(lt4,1)=-str_idx2
                              if (str_jdg2==0) then
                                 if (lord_tmp2>=vmin .and. lord_tmp2<=vmax) then
                                    exit
                                 endif
                              elseif (str_jdg2==1) then
                                 if (lord_tmp2<=vmin .or. lord_tmp2>=vmax) then
                                    exit
                                 endif
                              endif
                              !print*,step_cur2,step2,step_len2
                              if (step_cur2==mod(step_cur2+step2-step_sta2,step_len2)+step_sta2) then
                                 print*,"warning",lidx_rel2,lord_tmp2
                                 print*,"warning",str_idx2,str_jdg2,vmin,vmax
                                 pause
                              endif
                              step_cur2=mod(step_cur2+step2-step_sta2,step_len2)+step_sta2
                           enddo
!=========================================================================================!
                           !print*,"lidx_rel",lidx_rel,"j",j
                           !print*,"str_idx",str_idx,"lidx_rel",lidx_rel,lord_tmp
                           !print*,"str_idx",str_idx2,"lidx_tmp2",lidx_tmp2,lord_tmp2
                           !print*,"-2.0-",int(vt/dxl)*dxl,int(vt/dxl)*dxl+1,int(vt/dxl)*dxl+2,int(vt/dxl)*dxl+3
                           !print*,"-2.1-",vertexlist(int(vt/dxl)*dxl:int(vt/dxl)*dxl+dxl-1,1)
                           !print*,"-2.2-",vertexlist(int(vt/dxl)*dxl:int(vt/dxl)*dxl+dxl-1,2)                           

                           phase2jdg(:,2)=vertexlist(mm_tmp*dxl:mm_tmp*dxl+dxl-1,5)
                           !print*,"org",phase2jdg(:,1)
                           !print*,"org",phase2jdg(:,2)
                           call phase2vex(op_typ,-str_idx,-str_idx2)
                           vex2record_org=vex2record
                           do k=0,dxl-1
                              if (phase2jdg(k,1)==-str_idx) then
                                 !print*,"bef",phase2jdg(k,1),phase2jdg(k,2)
                                 phase2jdg(k,2)=phase2jdg(k,2)*dphase
                                 !print*,"aft",phase2jdg(k,1),phase2jdg(k,2)
                              elseif (phase2jdg(k,1)==-str_idx2) then
                                 !print*,"bef",phase2jdg(k,1),phase2jdg(k,2)
                                 phase2jdg(k,2)=phase2jdg(k,2)*jdgstring(str_idx2,7)
                                 !print*,"aft",phase2jdg(k,1),phase2jdg(k,2)
                              endif
                           enddo
                           !print*,"---------",dphase,strtab(str_idx2,5)
                           !print*,"nxt",phase2jdg(:,1)
                           !print*,"nxt",phase2jdg(:,2)
                           call phase2vex(op_typ,-str_idx,-str_idx2)
                           vex2record_nxt=vex2record

                           strstrel(str_idx,str_idx2,0)=strstrel(str_idx,str_idx2,0)+1
                           strstrel(str_idx,str_idx2,vex2record)=strstrel(str_idx,str_idx2,vex2record_nxt)+1

                           !print*,jdgstring(str_idx,2),jdgstring(str_idx2,2),op_typ+dphase*4+1
                           nextst(jdgstring(str_idx,2),jdgstring(str_idx2,2),0)=&
                           &nextst(jdgstring(str_idx,2),jdgstring(str_idx2,2),0)+1
                           nextst(jdgstring(str_idx,2),jdgstring(str_idx2,2),vex2record_nxt)=&
                           &nextst(jdgstring(str_idx,2),jdgstring(str_idx2,2),vex2record_nxt)+1

                           if (jdgstring(str_idx,2)>0d0 .and. jdgstring(str_idx2,2)>0d0) then
                              st1=int((midlstate(jdgstring(str_idx,2),0)+1)/2)
                              st2=int((midlstate(jdgstring(str_idx2,2),0)+1)/2)
                              !print*,"nxt 4",vex2record_nxt,st1,st2
                              widx1_nxt=vex2weight(vex2record_nxt,st1,st2,1)
                              widx2_nxt=vex2weight(vex2record_nxt,st1,st2,2)
                              !print*,"nxt 4",widx1_nxt,widx2_nxt
                              opstatenum(0,widx2_nxt,2)=&
                              &opstatenum(0,widx2_nxt,2)+2d0
                              opstatenum(widx1_nxt,widx2_nxt,2)=&
                              &opstatenum(widx1_nxt,widx2_nxt,2)+2d0
                              !print*,"nxt 4",vex2record_nxt,st1,st2
                              !print*,"nxt 4",widx1_nxt,widx2_nxt,midlstate(jdgstring(str_idx,2),0)*dphase,&
                              !&midlstate(jdgstring(str_idx2,2),0)*strtab(str_idx2,5)
                              !print*,"nxt 4","================================"
                              if (weight(widx1_nxt,widx2_nxt)==0d0) then
                              !   !print*,op_typ,statebw,midlstate(jdgstring(str_idx,2),0),midlstate(jdgstring(str_idx2,2),0)
                              !   !print*,jdgstring(str_idx,2),jdgstring(str_idx2,2)
                                 print*,"error nxt 4"
                                 pause
                              endif                              
                           endif

                           if (jdgstring(str_idx,1)>0d0 .and. jdgstring(str_idx2,1)>0d0) then
                              st1=int((loopstate(jdgstring(str_idx,1),0)+1)/2)
                              st2=int((loopstate(jdgstring(str_idx2,1),0)+1)/2)
                              !print*,"org 4",vex2record_nxt,st1,st2
                              widx1_org=vex2weight(vex2record_org,st1,st2,1)
                              widx2_org=vex2weight(vex2record_org,st1,st2,2)
                              !print*,"org 4",widx1_org,widx2_org
                              opstatenum(0,widx2_org,1)=&
                              &opstatenum(0,widx2_org,1)+2d0
                              opstatenum(widx1_org,widx2_org,1)=&
                              &opstatenum(widx1_org,widx2_org,1)+2d0
                              !print*,"org 4",vex2record_org,st1,st2
                              !print*,"org 4",widx1_org,widx2_org,loopstate(jdgstring(str_idx,1),0),&
                              !&loopstate(jdgstring(str_idx2,1),0)
                              !print*,"org 4","================================"
                              if (weight(widx1_org,widx2_org)==0d0) then
                                 !print*,op_typ,statebw,loopstate(jdgstring(str_idx,1),0),loopstate(jdgstring(str_idx2,1),0)
                                 !print*,jdgstring(str_idx,1),jdgstring(str_idx2,1)
                                 print*,"error org 4"
                                 pause
                              endif
                           endif

                           if (widx1_nxt/=widx1_org .or. widx2_nxt/=widx2_org) then
                              print*,"nxt err2",widx1_nxt,widx2_nxt,vex2record_nxt,&
                              &int((midlstate(jdgstring(str_idx,2),0)+1)/2),int((midlstate(jdgstring(str_idx2,2),0)+1)/2),&
                              &jdgstring(str_idx,2),jdgstring(str_idx2,2)
                              print*,"org err2",widx1_org,widx2_org,vex2record_org,&
                              &int((loopstate(jdgstring(str_idx,1),0)+1)/2),int((loopstate(jdgstring(str_idx2,1),0)+1)/2),&
                              &jdgstring(str_idx,1),jdgstring(str_idx2,1)
                              print*,jdgstring(:,2)
                              print*,jdgstring(:,1)
                              pause
                           endif

                        endif   
                        !if (step_cur>step_end) then
                           !exit
                        !endif
                     enddo
                  endif
                  !print*,"-------------------------------------------"
                  if (sum(opstatenum(0,:,2))/2-sum(opstatenum(0,:,1))/2/=crsign) then
                     print*,"sum err 1",sum(opstatenum(0,:,2)),sum(opstatenum(0,:,1)),crsign
                     pause
                  endif

               enddo
            endif
         endif
      enddo
      !print*,"o op"
      dprob=1d0
      !print*,"lty_t",ltyp_org,ltyp_nxt,crsign
      !print*,"opnum1",int(opstatenum(0,:,1)/2d0),mod(opstatenum(0,:,1),2)
      !print*,"opnum2",int(opstatenum(0,:,2)/2d0),mod(opstatenum(0,:,2),2)
      opstatenum(:,:,1)=int(opstatenum(:,:,1)/2d0)
      opstatenum(:,:,2)=int(opstatenum(:,:,2)/2d0)
      !if (sum(mod(opstatenum(0,:,1),2))/=0 .or. sum(mod(opstatenum(0,:,2),2))/=0) then
      !   pause
      !endif
      if (sum(opstatenum(0,:,2))-sum(opstatenum(0,:,1))/=crsign) then
         print*,"sum err",sum(opstatenum(0,:,2)),sum(opstatenum(0,:,1)),crsign
         pause
      endif
      do i=0,2
         do j=1,3
            !print*,j,i,opstatenum(j,i,:)
            if (opstatenum(j,i,1)/=0d0 .or. opstatenum(j,i,2)/=0d0) then
               if (weight(j,i)/=0d0) then
                  if (opstatenum(j,i,2)==opstatenum(j,i,1)) then
                  else
                     !print*,"weight",i,j,weight(j,i),opstatenum(j,i,2),opstatenum(j,i,1)
                     !print*,"weight",opstatenum(j,i,2),opstatenum(j,i,1),opstatenum(j,i,2)-opstatenum(j,i,1)
                     dprob=dprob*(weight(j,i))**(opstatenum(j,i,2)-opstatenum(j,i,1)) 
                  endif   
               elseif (weight(j,i)==0d0) then
                  !print*,"no-weight",opstatenum(j,i,2),opstatenum(j,i,1),opstatenum(j,i,2)-opstatenum(j,i,1)
                  dprob=0d0 
                  exit       
               endif
            endif
         enddo
      enddo
      !print*,"dprob",dprob
   else
      !print*,"skip"
   endif
   if (sprob/=dprob) then
      print*,"prob check",sprob,dprob
      pause
   endif

   !print*,"9999999999999999999999999999999999999999999"
   if (dprob/=0d0) then
      do i=1,dxl
         if (strtab(i,0)==0) then
            exit
         else
            lidx_rel=strtab(i,0)
            if ( lidx_rel>0 ) then
               !print*,"relating loop basis",lidx_rel
               do j=1, loopnummax
                  if (loopoprecord(lidx_rel,j,0)==0d0) then
                      cycle
                  else
                     check(:)=0d0
                     step_sta=strtab(loopstate(lidx_rel,2),1)
                     step_end=strtab(loopstate(lidx_rel,2),2)
                     step_len=(step_end-step_sta)+1
                     step_cur=step_sta
                     loadtyp=0d0
                     do 
                        step_par=mod(step_cur-step_sta+1,step_len)+step_sta
                        if (headsort(step_par,5)/=headsort(step_cur,5)) then
                           step_par=step_end
                           step=1
                           loadtyp=1d0
                           if (headsort(step_par,5)/=headsort(step_cur,5)) then
                           endif
                        else
                           step=2
                        endif
                        str_idx=headsort(step_cur,5)
                       str_jdg=mod(jdgstring(str_idx,0),2)
                       vmin=jdgstring(str_idx,3)
                       vmax=jdgstring(str_idx,4)
                     
                        if (loopstate(j,2)==0d0) then 
                          check(:)=check(:)+strgop(str_idx,j,:)
                        else
                           step_sta2=strtab(loopstate(j,2),1)
                           step_end2=strtab(loopstate(j,2),2)
                           step_len2=(step_end2-step_sta2)+1
                           step_cur2=step_sta2
                           loadtyp2=0d0
!=========================================================================================!
                           do 
                              step_par2=mod(step_cur2-step_sta2+1,step_len2)+step_sta2
                              if (headsort(step_par2,5)/=headsort(step_cur2,5)) then
                                 step_par2=step_end2
                                 step2=1
                                 loadtyp2=1d0
                                 if (headsort(step_par2,5)/=headsort(step_cur2,5)) then
                                 endif
                              else
                                 step2=2
                              endif
                              str_idx2=headsort(step_cur2,5)
                              str_jdg2=mod(jdgstring(str_idx2,0),2)
                              vmin=jdgstring(str_idx2,3)
                              vmax=jdgstring(str_idx2,4)
                              check(:)=check(:)+strstrel(str_idx,str_idx2,:)
                              !print*,"strstrel",strstrel(str_idx,str_idx2,:)
                              step_cur2=step_cur2+step2
                              if (step_cur2>=step_end2-loadtyp2) then
                                 exit
                              endif
                           enddo
                        endif

                        !print*,"s1",str_idx,lidx_rel,j
                        !print*,"s2",strgop(str_idx,j,:)
                        !print*,"1",step_cur,step_end,step_end-loadtyp
                        step_cur=step_cur+step
                        !print*,"2",step_cur,step_end,step_end-loadtyp
                        if (step_cur>=step_end-loadtyp) then
                           exit
                        endif
                     enddo
                     if (check(0)/=loopoprecord(lidx_rel,j,0)) then
                        print*,"not match",loopstate(j,2)==0d0
                        print*,"not match",check(0),loopoprecord(lidx_rel,j,0),lidx_rel,j
                        print*,"not match",check(:)
                        print*,"not match",loopop(lidx_rel,j,:)
                        print*,"not match",loopoprecord(lidx_rel,j,:)
                        pause
                     endif
                  endif
               enddo
            endif
         endif
      enddo
   endif
   !print*,"6666666666666666666666666666666666666666666"

   do i=0,dxl-1
      if (midstring(i,1)>0) then
         loopstate(midstring(i,1),2)=0
      endif
   enddo
   !print*,"finish string"
end subroutine initstringrelationtab
!==================================================!

!==================================================!
subroutine addoperator(mm_pos,b_pos,rank)
   use configuration
   use vertexupdate
   implicit none
   integer :: i,j,k,mm_pos,b_pos,s0,s1,leg_pos,up_pos,rank
   integer :: vt

   loopnumper=loopnumber
   !print*,"loop",loopnumber,loopnumper
   !print*,"start add",loopnum1,sum(loopstate(:,1))
   call gencurstateop(mm_pos,b_pos,1)
   !print*,"start add1",loopnum1,sum(loopstate(:,1))
   call updateoporder(mm_pos,1)
   !print*,"start add2",loopnum1,sum(loopstate(:,1))
   call initchgloopidx(mm_pos,b_pos,1)
   !print*,"gennet",loopnum1,sum(loopstate(:,1))

   do k=1,mxl
      s0=bsites(k,b_pos)
      if (s0==-1) cycle
      do i=0,1
         leg_pos=k-1+i*mxl
         vt=mm_pos*dxl+leg_pos
         sugloopidx(leg_pos)=vt
         !print*,vt,vertexlist(vt,2),custstateop(s0),frststateop(s0)
         if (vertexlist(vt,3)==2) then 
            !print*,vertexlist(vt,2),vertexlist(vt,3)
            !print*,"add"
            !print*,"--------------------------------"
            call markloop(vt,leg_pos,0,rank)
         endif
      enddo
   enddo
   !print*,"+++++++++++++++++++++++++++++++++add",ltyp_org,ltyp_nxt
   !print*,nxtloopidx(:)
   !print*,vertexlist(sugloopidx(0),1),vertexlist(sugloopidx(1),1)&
   !&,vertexlist(sugloopidx(2),1),vertexlist(sugloopidx(3),1)
   !do i=0,dxl-1
   !   if (nxtloopidx(i)/=vertexlist(sugloopidx(i),1)) then
   !      print*,"error",i,nxtloopidx(i),sugloopidx(i)
   !      pause
   !   endif
   !enddo

   !print*,"check"
   if (dloop/=loopnumber-loopnumper) then
      print*,"loopnumber_error_add",loopnumber-loopnumper,dloop,loopnumber,loopnumper,nn
      print*,"loopnumber_error_add",loopnumber-loopnumper,dloop,loopnumt,loopnum1,loopnum0
      pause
   endif
   call recordlooprelate(-3,-1,-1,-1)
   !print*,"markloop"

end subroutine addoperator
!==================================================!

!==================================================!
subroutine remoperator(mm_pos,b_pos,rank)
   use configuration
   use vertexupdate
   implicit none
   integer :: i,j,k,mm_pos,b_pos,s0,s1,leg_pos,up_pos,rank
   integer :: vt

   loopnumper=loopnumber
   !print*,"loop",loopnumber,loopnumper,loopnum1,loopnum0
   !print*,"start rem",loopnum1,sum(loopstate(:,1))
   call gencurstateop(mm_pos,b_pos,-1)
   !print*,"start rem1",loopnum1,sum(loopstate(:,1))
   call updateoporder(mm_pos,-1)
   !print*,"start rem2",loopnum1,sum(loopstate(:,1))
   call initchgloopidx(mm_pos,b_pos,-1)
   !print*,"gennet",loopnum1,sum(loopstate(:,1))

   do k=1,mxl
      s0=bsites(k,b_pos)
      if (s0==-1) cycle
      do i=0,1
         leg_pos=k-1+i*mxl
         vt=mm_pos*dxl+leg_pos
         !print*,vertexlist_map(vt),vertexlist(vertexlist_map(vt),2),custstateop(s0),frststateop(s0)
         sugloopidx(leg_pos)=vertexlist_map(vt)
         if (vertexlist_map(vt)/=-1 .and. vertexlist(vertexlist_map(vt),3)==2) then 
            !print*,vertexlist(vt,2)
            !print*,"rem"
            !print*,"--------------------------------"
            call markloop(vertexlist_map(vt),leg_pos,0,rank)
         endif
         vertexlist_map(vt)=-1
      enddo
   enddo
   !print*,"+++++++++++++++++++++++++++++++++rem",ltyp_org,ltyp_nxt
   !print*,nxtloopidx(:)
   !print*,vertexlist(sugloopidx(0),1),vertexlist(sugloopidx(1),1)&
   !&,vertexlist(sugloopidx(2),1),vertexlist(sugloopidx(3),1)
   !do i=0,dxl-1
   !   if (nxtloopidx(i)/=vertexlist(sugloopidx(i),1)) then
   !      print*,"error",i,nxtloopidx(i),sugloopidx(i)
   !      pause
   !   endif
   !enddo

   if (dloop/=loopnumber-loopnumper) then
      print*,"loopnumber_error_rem",loopnumber-loopnumper,dloop,loopnumber,loopnumper
      print*,"loopnumber_error_rem",loopnumber-loopnumper,dloop,loopnumt,loopnum1,loopnum0
      pause
   endif
   call recordlooprelate(-3,-1,-1,-1)
   !print*,"markloop"

end subroutine remoperator
!==================================================!


!==================================================!
subroutine rwroperator(mm_pos,b_pos,rank)
   use configuration
   use vertexupdate
   implicit none
   integer :: i,j,k,mm_pos,b_pos,s0,s1,leg_pos,up_pos,rank
   integer :: vt

   loopnumper=loopnumber
   !print*,"loop",loopnumber,loopnumper
   !print*,"start chg1",loopnum1,sum(loopstate(:,1))
   call gencurstateop(mm_pos,b_pos,2)
   !print*,"start chg2",loopnum1,sum(loopstate(:,1))
   call initchgloopidx(mm_pos,b_pos,0)

   do k=1,mxl
      s0=bsites(k,b_pos)
      if (s0==-1) cycle
      do i=0,1
         leg_pos=k-1+i*mxl
         vt=mm_pos*dxl+leg_pos
         sugloopidx(leg_pos)=vt
         !print*,vt,vertexlist(vt,2),custstateop(s0),frststateop(s0)
         if (vertexlist(vt,3)==2) then 
            !print*,vertexlist(vt,2)
            !print*,"rwr"
            !print*,"--------------------------------"
            call markloop(vt,leg_pos,0,rank)
         endif
      enddo
   enddo
   !print*,"+++++++++++++++++++++++++++++++++rwr",ltyp_org,ltyp_nxt
   !print*,nxtloopidx(:)
   !print*,vertexlist(sugloopidx(0),1),vertexlist(sugloopidx(1),1)&
   !&,vertexlist(sugloopidx(2),1),vertexlist(sugloopidx(3),1)
   !do i=0,dxl-1
   !   if (nxtloopidx(i)/=vertexlist(sugloopidx(i),1)) then
   !      print*,"error",i,nxtloopidx(i),sugloopidx(i)
   !      pause
   !   endif
   !enddo

   !print*,"check"
   if (dloop/=loopnumber-loopnumper) then
      print*,"loopnumber_error_rwr",loopnumber-loopnumper,dloop,loopnumber,loopnumper,nn
      print*,"loopnumber_error_rwr",loopnumber-loopnumper,dloop,loopnumt,loopnum1,loopnum0
      pause
   endif
   call recordlooprelate(-3,-1,-1,-1)
   !print*,"markloop"

end subroutine rwroperator
!==================================================!
subroutine gen_nextloop_idx(looptt)
   use configuration
   use vertexupdate
   integer :: i,j,k,l
   integer :: looptt

   if (rebootnum/=0)  then
      loopnumt=rebootloop(1)
      if (rebootnum/=1) then
         rebootloop(1:rebootnum-1)=rebootloop(2:rebootnum)
         rebootloop(rebootnum)=0d0
         rebootnum=rebootnum-1
      else
         rebootloop(1)=0
         rebootnum=rebootnum-1
      endif
   else
      loopnumt=loopnum1+1+looptt
   endif
   do j=0, chgloopnum
      if (chgloopidx(j,1)==loopnumt) then
         exit
      elseif (chgloopidx(j,1)==-1) then
         chgloopidx(chgloopnum,1)=loopnumt
         chgloopnum=chgloopnum+1
         exit
      endif
   enddo
   if (loopnumt==0) then
      print*,loopnumt
      pause
   endif
   loopnummax=max(loopnummax,loopnumt)
   !print*,"gen_nextloop_idx"
   call recordlooprelate(-1,-1,-1,-1)
   !print*,"gen_nextloop_idx"

end subroutine gen_nextloop_idx
!==================================================!
!==================================================!
subroutine initchgloopidx(mm_pos,b_pos,opsign)
   use configuration
   use vertexupdate
   integer :: i,j,k,l,stridx0,stridx1,stridx2,stridx3,vt
   integer :: mm_tmp,lidx_rel,lord_tmp,lidx_tmp,leg_tmp,ltyp_tmp,leg_tmp2
   integer :: mm_pos,b_pos,leg0,leg1,lrtmp,nrtmp,looptt,opsign,sublooplegth
   integer, allocatable :: visitab(:)
   integer, allocatable :: rebootloop_copy(:)
   
   nxtloopidx(:)=0d0
   allocate(visitab(0:dxl-1))
   visitab(:)=0d0
   nxtidx(:)=-1d0

   looptt=0d0
   do i=0,dxl-1
      vt=mm_pos*dxl+i
      !print*,"midstring(i,5)",midstring(i,5),nxtloopidx(i),nxtidx(midstring(i,5))
      if (nxtidx(midstring(i,5))==-1d0) then
         if (opsign==1d0 .or. opsign==0d0) then
            if (vertexlist(vt,3)==2d0) then
               call gen_nextloop_idx(looptt)
               looptt=looptt+1
               nxtidx(midstring(i,5))=loopnumt
            endif
         elseif (opsign==-1d0) then
            if (vertexlist_map(vt)/=-1 .and. vertexlist(vertexlist_map(vt),3)==2) then 
               call gen_nextloop_idx(looptt)
               looptt=looptt+1
               nxtidx(midstring(i,5))=loopnumt
            else
               nxtidx(midstring(i,5))=0d0
            endif
         endif         
         nxtloopidx(i)=nxtidx(midstring(i,5))
      else
         nxtloopidx(i)=nxtidx(midstring(i,5))
      endif
   enddo

   !print*,"check init","loopnummax",loopnummax
   !print*,"check init",chgloopidx(:,1)
   !print*,"check",loopnum1,sum(loopstate(:,1))

   do i=1,dxl
      !print*,i,"check",loopnum1,sum(loopstate(:,1))
      if (jdgstring(i,0)/=-1) then
         !print*,"nxt",i,jdgstring(i,2),nxtidx(jdgstring(i,2))
         if (nxtidx(jdgstring(i,2))>0d0) then
            loopstate(nxtidx(jdgstring(i,2)),0)=midlstate(jdgstring(i,2),0)
            loopstate(nxtidx(jdgstring(i,2)),2)=2d0*jdgstring(i,2)
         endif
         jdgstring(i,2)=nxtidx(jdgstring(i,2))
         !print*,i,jdgstring(i,2)
      else
         exit
      endif
   enddo
   !print*,"check",loopnum1,sum(loopstate(:,1))
   !print*,"check",chgloopidx(:,1)

   do i=0,chgloopnum
      if (chgloopidx(i,1)>0) then
         lidx_rel=chgloopidx(i,1)
         !print*,"check init",lidx_rel
         !print*,"reset loop basis",lidx_rel,loopnum1,sum(loopstate(:,1))
         if (loopstate(lidx_rel,2)==0d0) then
            loopstate(lidx_rel,2)=loopstate(lidx_rel,2)+1
         endif
         if (lidx_rel>0d0) then
            do j=1, loopnummax
               if (loopoprecord(lidx_rel,j,0)==0d0) then
                  !print*,"check init",lidx_rel,j
                  cycle
               else
                  !print*,"check init",lidx_rel,j,loopoprecord(lidx_rel,j,0),loopoprecord(j,lidx_rel,0)
                  do l=1,loopoprecord(lidx_rel,j,0)
                     loopoprecord(lidx_rel,j,l)=0d0
                  enddo
                  loopoprecord(lidx_rel,j,0)=0d0
                  loopop(lidx_rel,j,:)=0d0
   
                  do l=1,loopoprecord(j,lidx_rel,0)
                     loopoprecord(j,lidx_rel,l)=0d0
                  enddo
                  loopoprecord(j,lidx_rel,0)=0d0
                  loopop(j,lidx_rel,:)=0d0
                  !print*,"check init",lidx_rel,j,loopoprecord(lidx_rel,j,0),loopoprecord(j,lidx_rel,0)
               endif
            enddo
         endif
      else
         exit
      endif
   enddo
   !do i=0,loopnum
   !   print*,"init chg",i,chgloopidx(i,1)
   !   if (chgloopidx(i,1)>0) then
   !      lidx_rel=chgloopidx(i,1)
   !      print*,"loopstate",lidx_rel,loopstate(lidx_rel)
   !      !print*,"reset loop basis",lidx_rel
   !      if (loopstate(lidx_rel)==0d0) then
   !         print*,"loopstate error",lidx_rel
   !         pause
   !      endif
   !   else
   !      exit
   !   endif
   !enddo

   !print*,"================================================"
   !print*,ltyp_org,ltyp_nxt,mm_pos
   !print*,nxtloopidx(:)
   !print*,"st0",midstring(:,0)!number,org_loop,string,nxt_loop
   !print*,"st1",midstring(:,1)!loop connect of org
   !print*,"st2",midstring(:,2)!loop order of org
   !print*,"st3",midstring(:,3)!loop connect of string
   !print*,"st4",midstring(:,4)!loop connecting of string
   !print*,"st5",midstring(:,5)!loop connect of next
   !print*,"------------------------------------------------"
   !print*,"jg0",jdgstring(:,0)!loop connect of string
   !print*,"jg1",jdgstring(:,1)!loop connect of org
   !print*,"jg2",jdgstring(:,2)!loop connect of next
   !print*,"jg3",jdgstring(:,3)
   !print*,"jg4",jdgstring(:,4)
   !print*,"chgloopidx",chgloopidx(:,1)
   !print*,"================================================"
   !pause

   !do i=1,chgloopnum
   !   if (chgloopidx(i,1)==-1) then
   !      exit
   !   else
   !      if (loopstate(chgloopidx(i,1))/=1d0) then
   !         print*,"warning2.1",chgloopidx(i,1),loopstate(chgloopidx(i,1)),loopnummax
   !         pause
   !      endif
   !   endif
   !enddo
   !if (sum(loopstate)/=chgloopnum) then
   !   print*,"warning2.2",sum(loopstate),chgloopnum
   !   pause
   !endif

end subroutine initchgloopidx
!==================================================!


subroutine markloop(vt,leg_pos,signal,rank)
   use configuration
   use vertexupdate
   implicit none

   integer :: i,j,k,signal,rank,flipsign,mmo,leg0,statechange,tmp,vt
   integer :: check,secount,frcount,leg_pos
   integer, allocatable :: opvisited(:)
   real(8), external :: ran
   real(8), external :: hf

   !loopnumt=nxtloopidx(mod(vt,dxl))

   !call gen_nextloop_idx(0)
   !loopnumt=loopnumt
   if (sum(loopstate(:,1))/=loopnum1) then
      pause
   endif
   loopnumt=nxtloopidx(leg_pos)
   loopstate(loopnumt,1)=1
   call recordlooprelate(-1,-1,-1,-1)
   Spm_measure_signal=signal
   loop_counter=0d0
   looporder=1

   !do i=1,dxl*mm-1
   !  if (vertexlist(i,1)==loopnumt) then
   !     print*,"mark_error",loopnumt,loopnum1,vertexlist(i,1),rebootnum
   !     print*,"mark_error",vertexlist(i,:)
   !     print*,rebootloop(:)
   !     pause
   !  endif
   !enddo

   v0=vt
   !print*,v0
   !statechange=int(sun*ran())
   statechange=1d0
   cc=statechange
   allocate(opvisited(nh))
   opvisited(:)=0d0

   v1=v0
   if (mod(v1,dxl)<mxl) then
      loopphase=1
   else
      loopphase=-1
   endif
   counter=1
   rep_counter=0d0
   if (mod(v0,dxl)>=mxl) then
      cc0=mod(int(sun)-cc,int(sun))          
   else
      cc0=mod(int(sun)+cc,int(sun))       
   endif

   do
      if ( v1==v0 .and. counter/=1 ) then
         !if (Spm_measure_signal==1) 
         !print*,"==============================loop1================================"
         if (statechange/=0d0) loop_counter=loop_counter+1
         loopnumber=loopnumber+1
         loopnum1=loopnum1+1
         !print*,loop_counter,rank
         exit
      endif

      i=v1/dxl
      op=opstring(i)
      b=op/4

      opvisited(tauscale(i))=opvisited(tauscale(i))+1

      if (v1<0) then
         print*,"v1 error",v1
         pause
      endif
      !if (Spm_measure_signal==1) print*,b,2*nn,v1,v0,rank

      if (b<=pbAc .and. sambon(bontab(b))<=nbs ) then
         if (mod(op,4)==0) then
            call strg(i,statechange)
         elseif(mod(op,4)==1) then
            call turn(i,statechange)
         elseif(mod(op,4)==2) then
            call jump(i,statechange)
         endif
      elseif (b>pbAc .and. b<=pbAc+pbBs &
         &.and. sambon(bontab(b))<=nbs .and. nnn==1d0) then
      elseif (b>pbAc+pbBs .and. b<=pbAc+pbBs+pbCn &
         &.and. sambon(bontab(b))<=nbs .and. nnn==1d0) then
      endif

      counter=counter+1
      !print*,"====",counter-1,rep_counter

      if ( v2==v0 ) then
         !if (Spm_measure_signal==1) 
         !print*,"==============================loop2================================"
         if (statechange/=0d0) loop_counter=loop_counter+1
         loopnumber=loopnumber+1
         loopnum1=loopnum1+1
         !vertexlist(v2,0)=ieor(vertexlist(v2,0),1)
         !vertexlist(vertexlist_map(v2),0)=ieor(vertexlist(vertexlist_map(v2),0),1)
         !print*,vertexlist(v2,0)
         !print*,vertexlist(vertexlist_map(v2),0)
         !print*,loop_counter,rank
         exit
      endif
   end do

   !frcount=0d0
   !secount=0d0
   !do i=1,nh
   !   if (opvisited(i)/=0d0) then
   !      if (opvisited(i)==1) then
   !         frcount=frcount+1
   !      elseif (opvisited(i)==2) then
   !         secount=secount+1
   !      else
   !         print*,"opvisited",opvisited(i)
   !      endif
   !      !print*,"sum",i,opvisited(i),int(v0/dxl),nh
   !   endif
   !enddo
   !print*,"loop end",counter-1,rep_counter,sum(opvisited),nh
   !print*,frcount,secount
   !if (secount/=rep_counter) then
      !print*,"error",rep_counter,frcount,secount
      !pause
   !else
      !print*,"crrect",rep_counter,secount      
   !endif
   !loopoprecord(loopnumt,:,0)=curoplp(:,0)
   check=0d0
   do i=1,loopnummax
      if (curoplp(i,0)/=0) then
         !check=check+1
         print*,i,curoplp(i,0),nh
      endif
   enddo
   !print*,"curoplp",curoplp(:,1)
   !if (sum(curoplp(:,1))/=frcount+2*secount) then
   !   print*,"more",loopnumt,i,curoplp(i,1),sum(curoplp(:,1)),frcount+2*secount,nh,frcount,secount
   !   pause
   !endif
   !if (check/=0) print*,"check loop touch",check,loopnumber,loopnummax,counter
   curoplp(:,0)=0d0
   !print*,"++++++++++++++++++++++++++++++++++++++++++++ one loop end"

end subroutine markloop
!==================================================!
subroutine recordlooprelate(mm_pos,v1t,v2t,ltyp)
   use configuration
   use vertexupdate
   use measurementdata
   implicit none
   integer :: crsign,size1,size2,mm_pos,ltyp,leg0t,legtt,sublooplegth
   integer :: v0t,vtt,v1t,v2t,i,j,loadsign,loadtmp,l
   integer :: mrkt,opodt,leg1,leg2,mrktst
   integer :: lidx0,lidx1,lord0,lord1,lidx_mid0,lidx_mid1
   integer :: opod0,opod1,mm_0,mm_1
   integer, allocatable :: visitab(:)
   integer, allocatable :: loopoprecord_copy(:,:,:)
   integer, allocatable :: strgoprecord_copy(:,:,:)
   integer, allocatable :: check(:)

   size1=size(loopoprecord,dim=3)
   size2=size(loopoprecord,dim=2)
   if (mm_pos==-2) then
      !print*,"check",mm_pos
      if (2*nh>=size1) then
         allocate(loopoprecord_copy(size2,size2,0:size1-1))
         loopoprecord_copy(:,:,:)=loopoprecord(:,:,:)
         deallocate(loopoprecord)
         allocate(loopoprecord(size2,size2,0:int(4d0*nh)))
         loopoprecord(:,:,:)=0d0
         loopoprecord(1:size2,1:size2,0:size1-1)=loopoprecord_copy(1:size2,1:size2,0:size1-1)
         deallocate(loopoprecord_copy)
      endif
      !print*,"check"
      !do i=0,dxl-1
      !   if (chgloopidx(i,1)==-1) then
      !      exit
      !   elseif (chgloopidx(i,1)==loopnumt) then
      !      chgloopidx(i,2)=chgloopidx(i,2)+1
      !      exit
      !   endif
      !enddo
      curoplp(:,:)=0d0

   elseif (mm_pos==-1) then
      !print*,"check",mm_pos
      if (loopnummax>=size2) then
         allocate(loopoprecord_copy(size1,size2,0:size1-1))
         loopoprecord_copy(:,:,:)=loopoprecord(:,:,:)
         deallocate(loopoprecord)
         allocate(loopoprecord(2*loopnummax,2*loopnummax,0:size1-1))
         loopoprecord(:,:,:)=0d0
         loopoprecord(1:size2,1:size2,0:size1-1)=loopoprecord_copy(1:size2,1:size2,0:size1-1)
         !print*,"loopnummax",mm_pos

         loopoprecord_copy(:,:,0:12)=loopop(:,:,:)
         deallocate(loopop)
         allocate(loopop(2*loopnummax,2*loopnummax,0:12))
         loopop(:,:,:)=0d0
         loopop(1:size2,1:size2,0:12)=loopoprecord_copy(1:size2,1:size2,0:12)
         !print*,"loopop",mm_pos


         allocate(strgoprecord_copy(dxl,size2,0:12))
         strgoprecord_copy(:,:,:)=strgop(:,:,:)
         deallocate(strgop)
         allocate(strgop(dxl,2*loopnummax,0:12))
         strgop(:,:,:)=0d0
         strgop(1:dxl,1:size2,0:12)=strgoprecord_copy(1:dxl,1:size2,0:12)

         strgoprecord_copy(:,:,:)=nextlp(:,:,:)
         deallocate(nextlp)
         allocate(nextlp(dxl,2*loopnummax,0:12))
         nextlp(:,:,:)=0d0
         nextlp(1:dxl,1:size2,0:12)=strgoprecord_copy(1:dxl,1:size2,0:12)
         !print*,"strgop",mm_pos

         strgoprecord_copy(1,:,0:2)=loopstate(:,0:2)
         deallocate(loopstate)
         allocate(loopstate(2*loopnummax,0:2))
         deallocate(updatestate)
         allocate(updatestate(2*loopnummax,2))
         deallocate(updatememory)
         allocate(updatememory(2*loopnummax,2))
         loopstate(:,0:2)=0d0
         loopstate(1:size2,0:2)=strgoprecord_copy(1,1:size2,0:2)
         deallocate(strgoprecord_copy)
         deallocate(curoplp)
         allocate(curoplp(2*loopnummax,0:dxl))
         !print*,"check"
      endif
      curoplp(:,:)=0d0
   elseif (mm_pos>=0d0) then
      !print*,"check",ltyp,mm_pos
      allocate(visitab(0:dxl-1))
      visitab(:)=0d0
      opodt=tauscale(mm_pos)
      leg1=mod(v1t,dxl)
      leg2=mod(v2t,dxl)
      visitab(leg1)=1
      visitab(leg2)=1
      vertexlist(v1t,3)=3
      vertexlist(v2t,3)=3
      loadsign=0d0
      loadtmp=0d0
      !lidx0=vertexlist(v1t,1)!old info
      !lord0=vertexlist(v1t,2)!old info
      do i=0, dxl-1
         if (visitab(i)==1) cycle
         v0t=mm_pos*dxl+i
         if (vertexlist_map(v0t)==-1) cycle
         leg0t=i
         legtt=linktable(i,ltyp)
         visitab(leg0t)=1
         visitab(legtt)=1
         vtt=mm_pos*dxl+legtt
         mrkt=vertexlist(v0t,1)
         loadsign=-1d0
         !print*,"mrkt",mrkt,chgloopidx(:,1)
         !print*,"mrkt",vertexlist(v0t,:)
         !print*,"mrkt",v0t,vtt,v1t,v2t,mm_pos
         !print*,"mrkt",leg1,leg2
         !print*,"mrkt",leg0t,legtt
         if (v0t==v1t .or. v0t==v2t) then
            cycle
         else
            !do j=0, chgloopnum
               !print*,"chgloopidx",chgloopidx(j,1)
               !if (mrkt==chgloopidx(j,1) .or. mrkt==0d0) then
            if (mrkt/=0d0) then
               if (loopstate(mrkt,2)==0d0) then
                  loadsign=0d0
                  loadtmp=loadtmp+1
                  !exit
               elseif (loopstate(mrkt,2)/=0d0) then
                  !print*,"::::::::",mrkt==chgloopidx(j,1),mrkt,vertexlist(v0t,4)
                  if (vertexlist(v0t,3)/=3) then
                     !print*,"vertexlist(j,4)==1",vertexlist(v0t,4)==1
                     loadsign=1d0
                     loadtmp=loadtmp+1
                    if (mrkt==vertexlist(v1t,1)) then
                       rep_counter=rep_counter+1
                     !print*,"test",mrkt,vertexlist(v1t,1),i,loopnumt
                        !pause
                  endif
                 else
                    !print*,"vertexlist(j,4)==1",vertexlist(v0t,4)==1
                     !rep_counter=rep_counter+1
                     loadsign=2d0
                  endif
                  !exit
               else
                  loadsign=-1d0       
               endif
            else
               loadsign=1d0
            endif
            !enddo
            if (loadsign==-1d0) then
               print*,"loadsign error",loadsign
            endif
            !print*,"-------------------------------------","loadsign",mrkt,curoplp(mrkt,0),loadsign
            !do l=1,loopnummax
            !   print*,curoplp(l,0)
            !enddo
            mrktst=mrkt
            call UpOperRelTab(v1t,v2t,v0t,vtt,loadsign,mm_pos)
            if (curoplp(mrkt,0)>2*nh) then
               print*,"curoplp",curoplp(mrkt,0),2*nh
               pause
            endif
            !print*,"-------------------------------------","curoplp(mrkt)",mrkt,curoplp(mrkt,0),loopnummax,size(curoplp,dim=1)
            !if (mrktst==0d0) then
            !   print*,mrkt,vertexlist(v0t,1)
            !   print*,v0
            !   print*,v0t,vtt
            !   print*,v1t,v2t
            !   pause
            !endif
         endif
         !print*,"mm_pos",mm_pos,int(v1t/dxl),int(v0/dxl),v0,v1t,v2t
         !print*,loadtmp,curoplp(mrkt,1),"mrkt",mrkt==vertexlist(v1t,1),vertexlist(v1t,1),&
         !&int(vertexlist(v1t,2)/2),nh
         !print*,mm_pos,vertexlist(mm_pos*dxl+0:mm_pos*dxl+dxl-1,1)
         !print*,chgloopidx(:,1)
         !print*,"rep_counter",rep_counter
      enddo  
      !print*,"+++++++++++++++++++++++++++++++++++"
      if (loadtmp>1) then
         print*,"cur_err",loadtmp,"mrkt",mrkt==vertexlist(v1t,1)
         pause
      endif       

   elseif (mm_pos==-3) then
      allocate(check(0:12))
      check(:)=0d0

      do i=0,chgloopnum
         if (chgloopidx(i,1)>0) then
            lidx0=chgloopidx(i,1)
            do j=1, loopnummax
               if (loopoprecord(j,lidx0,0)==0d0) then
                  cycle
               else
                  if (mod(loopstate(lidx0,2),2)==0d0) then
                     lidx_mid0=int(loopstate(lidx0,2)/2)
                     if (loopstate(j,2)==0d0 .or. mod(loopstate(lidx0,2),2)==1d0) then
                        lidx_mid1=j
                        !print*,"case 1",lidx_mid0,lidx_mid1,lidx0,j,loopstate(j,2)==0d0,mod(loopstate(lidx0,2),2)==1d0
                        check(:)=nextlp(lidx_mid0,lidx_mid1,:)
                        if (check(0)/=loopop(lidx0,j,0)) then
                           print*,lidx0,j,lidx_mid0,lidx_mid1
                           print*,"WARNING NOT MATCH1",check(:)
                           print*,"WARNING NOT MATCH1",loopop(lidx0,j,:)
                           pause
                        endif
                     elseif (loopstate(j,2)/=0d0 .and. mod(loopstate(lidx0,2),2)==0d0) then
                        lidx_mid1=int(loopstate(j,2)/2)
                        !print*,"case 2",lidx_mid0,lidx_mid1,lidx0,j
                        check(:)=nextst(lidx_mid0,lidx_mid1,:)
                        !print*,nextst(1,1,0),nextst(1,2,0),nextst(2,1,0),nextst(2,2,0)
                        if (check(0)/=loopop(lidx0,j,0)) then
                           print*,lidx0,j
                           print*,"WARNING NOT MATCH2",check(:)
                           print*,"WARNING NOT MATCH2",loopop(lidx0,j,:)
                           pause
                        endif
                     endif
                  endif
               endif
            enddo
         endif
      enddo

      do i=0,chgloopnum
         if (chgloopidx(i,1)>0) then
            lidx0=chgloopidx(i,1)
            do j=1, loopnummax
               if (loopoprecord(j,lidx0,0)==0d0) then
                  cycle
               else
                  if (loopoprecord(j,lidx0,0)/=loopoprecord(lidx0,j,0)) then
                     print*,"WARNING",lidx0,j,loopoprecord(j,lidx0,0),loopoprecord(lidx0,j,0)
                     pause
                  endif
                  !if (loopoprecord(j,lidx0,loopoprecord(lidx0,j,0)+1)/=0d0) then
                  !  print*,"WARNING3"
                  !   pause
                  !endif
                  if (loopop(j,lidx0,0)==1 .and. loopoprecord(j,lidx0,0)>1) then
                     sublooplegth=loopoprecord(j,lidx0,0)
                     allocate(visitab(sublooplegth))
                     visitab(1:sublooplegth)=loopoprecord(j,lidx0,1:sublooplegth)
                     call quick_sort_int(visitab(1:sublooplegth),sublooplegth,1,sublooplegth) 
                     loopoprecord(j,lidx0,1:sublooplegth)=visitab(1:sublooplegth)    
                     deallocate(visitab)
                     !print*,loopoprecord(j,lidx0,0:sublooplegth+1)
                     !print*,"loopoprecord(j,lidx0,0:sublooplegth+1)"
                  endif        
               endif
            enddo
            loopstate(lidx0,2)=0d0
         else
            exit
         endif
      enddo
   endif
   

end subroutine recordlooprelate
!==================================================!

!==================================================!
subroutine UpOperRelTab(v1t,v2t,v0t,vtt,lsign,mm_pos)
   use configuration
   use vertexupdate
   use measurementdata
   implicit none
   integer :: i,j,k,l,mm_pos,op_typ,lsign,b_tmp,sitmp
   integer :: lidx0,lord0,legt0,phas0
   integer :: lidx1,lord1,legt1,phas1
   integer :: lidx2,lord2,legt2,phas2
   integer :: lidxt,lordt,legtt,phast
   integer :: str1_dir,str1_idx,str0_dir,str0_idx,dphase,phasin
   integer :: v0t,vtt,v1t,v2t,legnum,lorg_idx,vin,op_typ_tmp

   lidx1=vertexlist(v1t,1)
   lord1=vertexlist(v1t,2)
   phas1=vertexlist(v1t,5)
   legt1=mod(v1t,dxl)
   phase2jdg(legt1,1)=lidx1
   phase2jdg(legt1,2)=phas1
   !=====================!
   lidx2=vertexlist(v2t,1)
   lord2=vertexlist(v2t,2)
   phas2=vertexlist(v2t,5)
   legt2=mod(v2t,dxl)
   phase2jdg(legt2,1)=lidx2
   phase2jdg(legt2,2)=phas2
   !=====================!
   lidx0=vertexlist(v0t,1)
   lord0=vertexlist(v0t,2)
   phas0=vertexlist(v0t,5)
   legt0=mod(v0t,dxl)
   phase2jdg(legt0,1)=lidx0
   phase2jdg(legt0,2)=phas0
   !=====================!
   lidxt=vertexlist(vtt,1)
   lordt=vertexlist(vtt,2)
   phast=vertexlist(vtt,5)
   legtt=mod(vtt,dxl)
   phase2jdg(legtt,1)=lidxt
   phase2jdg(legtt,2)=phast
   !=====================!
   op_typ=mod(opstring(mm_pos),4)

   if (mod(v0t,2)==1) then
      vin=v0t
      phasin=phas0
   elseif (mod(vtt,2)==1) then
      vin=vtt
      phasin=phast
   endif

   dphase=abs(phas1-phasin)
   opdhase(mm_pos)=dphase
   !print*,lsign,loopnumt,lidx0

   !print*,vertexlist(v0t,1),loopstate(vertexlist(v0t,1))
   !print*,"-2.1-",vertexlist(mm_pos*dxl:mm_pos*dxl+dxl-1,1)
   !print*,"-2.2-",vertexlist(mm_pos*dxl:mm_pos*dxl+dxl-1,2)
   !print*,"-2.4-",vertexlist(mm_pos*dxl:mm_pos*dxl+dxl-1,4)
   !print*,"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" 

   if (lsign==0) then ! one changing  
      !print*,"lsign==0",loopnumt,lidx0,op_typ,loopoprecord(lidx0,loopnumt,0),loopoprecord(loopnumt,lidx0,0)
      if (loopoprecord(lidx0,loopnumt,0)/=loopoprecord(loopnumt,lidx0,0)) then
         print*,"neq",loopoprecord(lidx0,loopnumt,0),loopoprecord(loopnumt,lidx0,0)
         pause
      endif

      loopoprecord(loopnumt,lidx0,0)=loopoprecord(loopnumt,lidx0,0)+1
      loopoprecord(loopnumt,lidx0,loopoprecord(loopnumt,lidx0,0))=v1t
      vertexlist(v0t,4)=v1t
      vertexlist(vtt,4)=v1t
      !print*,"checka"
      !print*,"checkb",loopoprecord(lidx0,loopnumt,0),size(loopoprecord,dim=3),nh,looporder/2,counter
      if (loopoprecord(lidx0,loopnumt,0)>counter) then
         print*,"checkb"
         pause
      endif
      loopoprecord(lidx0,loopnumt,0)=loopoprecord(lidx0,loopnumt,0)+1
      loopoprecord(lidx0,loopnumt,loopoprecord(lidx0,loopnumt,0))=vin
      vertexlist(v1t,4)=vin
      vertexlist(v2t,4)=vin

      vertexlist(v1t,3)=1
      vertexlist(v2t,3)=1

      call phase2vex(op_typ,lidx0,loopnumt)
      loopop(lidx0,loopnumt,0)=loopop(lidx0,loopnumt,0)+1
      loopop(lidx0,loopnumt,vex2record)=loopop(lidx0,loopnumt,vex2record)+1
      call phase2vex(op_typ,loopnumt,lidx0)
      loopop(loopnumt,lidx0,0)=loopop(loopnumt,lidx0,0)+1
      loopop(loopnumt,lidx0,vex2record)=loopop(loopnumt,lidx0,vex2record)+1
      !print*,"checkc"
      !print*,"lsign==0",loopnumt,lidx0,op_typ,mm_pos,loopoprecord(lidx0,loopnumt,0),loopoprecord(loopnumt,lidx0,0)
   elseif (lsign==1) then ! two changing one changing one will change
      sitmp=-1
      if (lidx0/=0d0) then
         do str0_idx=1,dxl
            if (jdgstring(str0_idx,0)==-1) then
               exit
            else
               if (jdgstring(str0_idx,1)==lidx0 .and. jdgstring(str0_idx,1)==lidxt) then
                  if (mod(jdgstring(str0_idx,0),2)==0d0) then
                     if (lord0>=jdgstring(str0_idx,3) .and. lord0<=jdgstring(str0_idx,4)) then
                        exit
                     endif
                  elseif (mod(jdgstring(str0_idx,0),2)==1d0) then
                     if (lord0<=jdgstring(str0_idx,3) .or. lord0>=jdgstring(str0_idx,4)) then
                        exit
                    endif
                  endif
               elseif (jdgstring(str0_idx,1)==lidx0 .or. jdgstring(str0_idx,1)==lidxt) then
                  op_typ_tmp=mod(opstring(mm_pos),4)
                  if (legt0==linktable(legtt,op_typ_tmp)) then
                     if (mod(jdgstring(str0_idx,0),2)==0d0) then
                        if (lord0>=jdgstring(str0_idx,3) .and. lord0<=jdgstring(str0_idx,4)) then
                           exit
                       endif
                     elseif (mod(jdgstring(str0_idx,0),2)==1d0) then
                        if (lord0<=jdgstring(str0_idx,3) .or. lord0>=jdgstring(str0_idx,4)) then
                           exit
                        endif
                     endif
                 else
                     print*,"unknow"
                    pause
                 endif
              endif
           endif
         enddo
      else
         b_tmp=int(opstring(mm_pos)/4d0)
         sitmp=mod(v0t,mxl)
         !print*,v0t
         !print*,"sitmp",sitmp,b_tmp,bsites(sitmp,b_tmp),mod(opstring(mm_pos),4)
         sitmp=bsites(sitmp+1,b_tmp)
         !print*,"sitmp",sitmp,b_tmp,mxl
         do str0_idx=1,dxl
            if (jdgstring(str0_idx,0)==-1) then
               exit
            else
               if (jdgstring(str0_idx,4)==sitmp .and. jdgstring(str0_idx,3)==0d0) then
                  exit
               endif
            endif
         enddo
      endif
      !print*,"---","ltyp",ltyp_org,ltyp_nxt
      !print*,"---","lsign",lsign,"lidx0",lidx0,"legt0",legt0,"lidxt",lidxt,"v0",v0
      !print*,"-0-",v0t,vtt,loopnumt,lidx0,lidxt,op_typ_tmp,legt0
      !print*,"-1-",mod(v0t,mxl),mm_pos,v0t,b_tmp
      !print*,"-2.0-",mm_pos*dxl,mm_pos*dxl+1,mm_pos*dxl+2,mm_pos*dxl+3
      !print*,"-2.1-",vertexlist(mm_pos*dxl:mm_pos*dxl+dxl-1,1)
      !print*,"-2.2-",vertexlist(mm_pos*dxl:mm_pos*dxl+dxl-1,2)
      !print*,"-2.4-",vertexlist(mm_pos*dxl:mm_pos*dxl+dxl-1,4)
      !print*,"-3-",chgloopidx(:,1),sum(loopstate)
      !print*,"-4-",jdgstring(str0_idx,2),str0_idx,sitmp
      !print*,"jdgstring2",jdgstring(:,2)
      lidx0=jdgstring(str0_idx,2)
      !print*,"lsign==1",loopnumt,lidx0,op_typ,loopoprecord(lidx0,loopnumt,0),loopoprecord(loopnumt,lidx0,0)
      if (lidx0>0) then
         loopoprecord(loopnumt,lidx0,0)=loopoprecord(loopnumt,lidx0,0)+1
         loopoprecord(loopnumt,lidx0,loopoprecord(loopnumt,lidx0,0))=v1t 
         vertexlist(v0t,4)=v1t
         vertexlist(vtt,4)=v1t
      else
         print*,"lidx0",lidx0
         pause
      endif  
      !print*,"lsign==1",loopnumt,lidx0,op_typ,mm_pos,loopoprecord(lidx0,loopnumt,0),loopoprecord(loopnumt,lidx0,0)
   elseif (lsign==2) then ! two changing one changing one changed
      !print*,"lsign==2",loopnumt,lidx0,op_typ,loopoprecord(lidx0,loopnumt,0),loopoprecord(loopnumt,lidx0,0) 
      loopoprecord(loopnumt,lidx0,0)=loopoprecord(loopnumt,lidx0,0)+1
      loopoprecord(loopnumt,lidx0,loopoprecord(loopnumt,lidx0,0))=v1t 
      vertexlist(v0t,4)=v1t
      vertexlist(vtt,4)=v1t

      vertexlist(v1t,3)=1
      vertexlist(v2t,3)=1
      vertexlist(v0t,3)=1
      vertexlist(vtt,3)=1

      call phase2vex(op_typ,lidx0,loopnumt)
      loopop(lidx0,loopnumt,0)=loopop(lidx0,loopnumt,0)+1
      loopop(lidx0,loopnumt,vex2record)=loopop(lidx0,loopnumt,vex2record)+1
      call phase2vex(op_typ,loopnumt,lidx0)
      loopop(loopnumt,lidx0,0)=loopop(loopnumt,lidx0,0)+1
      loopop(loopnumt,lidx0,vex2record)=loopop(loopnumt,lidx0,vex2record)+1
      !print*,"lsign==2",loopnumt,lidx0,op_typ,mm_pos,loopoprecord(lidx0,loopnumt,0),loopoprecord(loopnumt,lidx0,0) 
   elseif (lsign==-1) then
      print*,"lsign",lsign
      pause
   endif
   !print*,"\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\" 

end subroutine UpOperRelTab
!==================================================!

!==================================================!
subroutine gencurstateop(i,b_pos,crsign)
   use configuration
   use measurementdata
   implicit none

   integer :: i,k,b_pos,s,mm_pos,vt0,vt1,vt2,vt3,crsign,up_pos,leg_pos,j
   integer :: mmt2,mmt3,rmcounter,stcounter,curchgmrk
   integer, allocatable :: curchgmrktab(:,:)

   mm_pos=i
   curchgmrk=0d0
   allocate(curchgmrktab(0:dxl-1,2))
   curchgmrktab(:,:)=0d0
   if (crsign==0) then
      !print*,"---",crsign,b_pos,opstring(mm_pos)/4
      do k=1, mxl
         s=bsites(k,b_pos)
         if (s==-1) cycle
         vt0=dxl*mm_pos+k-1
         vt1=dxl*mm_pos+k-1+mxl
         if (vertexlist_map(custstateop(s))/=vt0) then
            print*,vertexlist_map(custstateop(s)),vt0,"map_error",mm_pos 
            pause
         endif
         vertexlist_map(custstateop(s))=vt0
         vertexlist_map(vt0)=custstateop(s)
         custstateop(s)=vt1
         curchgmrktab(k-1,1)=vertexlist(vt0,1)
         curchgmrktab(k-1+mxl*(ieor(up_pos,1)),1)=vertexlist(vt1,1)
      enddo
   elseif (crsign==1) then
      !print*,crsign
      do k=1,mxl
         s=bsites(k,b_pos)
         if (s==-1) cycle
         up_pos=0
         leg_pos=k-1+mxl*up_pos
         vt0=mm_pos*dxl+leg_pos
         vt1=mm_pos*dxl+k-1+mxl*(ieor(up_pos,1))
         if (frststateop(s)==-1) then
            frststateop(s)=vt0
            vertexlist_map(vt0)=vt1
            vertexlist_map(vt1)=vt0
            custstateop(s)=vt1
            vertexlist(vt0,3)=2
            vertexlist(vt1,3)=2
            curchgmrktab(k-1,1)=vertexlist(vt0,1)
            curchgmrktab(k-1+mxl*(ieor(up_pos,1)),1)=vertexlist(vt1,1)
            loopnum0=loopnum0-1d0
            loopnumber=loopnumber-1d0
         else
            vt3=custstateop(s)
            vt2=vertexlist_map(custstateop(s))
            !print*,"gen+=========================",loopnumber,loopnumper
            call loadlooptable(vertexlist(vt2,1))
            !print*,"gen00000000000000000000000000",loopnumber,loopnumper
            call loadlooptable(vertexlist(vt3,1))
            !print*,"gen--------------------------",loopnumber,loopnumper

            vertexlist_map(vt3)=vt0 
            vertexlist_map(vt0)=vt3
            vertexlist(vt0,1:2)=vertexlist(vt3,1:2)

            vertexlist_map(vt2)=vt1
            vertexlist_map(vt1)=vt2
            vertexlist(vt1,1:2)=vertexlist(vt2,1:2)

            curchgmrktab(k-1,1)=vertexlist(vt0,1)
            curchgmrktab(k-1+mxl*(ieor(up_pos,1)),1)=vertexlist(vt1,1)

            if (frststateop(s)==vt2) then
               mmt2=int(vt2/dxl)
               mmt3=int(vt3/dxl)
               if (mmt2>mm_pos) then
                  frststateop(s)=vt0
               endif
            endif
            custstateop(s)=vt1
            vertexlist(vt0,3)=2
            vertexlist(vt1,3)=2
            vertexlist(vt2,3)=2
            vertexlist(vt3,3)=2
         endif
         !print*,state(s)
      enddo

   elseif (crsign==2) then
      !print*,crsign
      do k=1, mxl
         !print*,"||||||||||||||||||||||||||||||||test||||||||||||||||||||",mm_pos
         s=bsites(k,b_pos)
         if (s==-1) cycle
         vt0=dxl*mm_pos+k-1
         vt1=dxl*mm_pos+k-1+mxl
         if (vertexlist_map(custstateop(s))/=vt0) then
            print*,vertexlist_map(custstateop(s)),vt0,"map_error2",mm_pos
            pause
         endif
         vt3=custstateop(s)
         if (vertexlist_map(vt3)/=vt0) then
            print*,"check",vertexlist_map(vt3)/=vt0,vertexlist_map(vt3),vt0
            pause
         endif
         vertexlist_map(vt3)=vt0
         vertexlist_map(vt0)=vt3
         custstateop(s)=vt1
         vt2=vertexlist_map(vt1)
         !print*,vt3,vt0,vt1,vt2
         call loadlooptable(vertexlist(vt2,1))
         call loadlooptable(vertexlist(vt3,1))
         vertexlist(vt0,3)=2
         vertexlist(vt1,3)=2
         vertexlist(vt2,3)=2
         vertexlist(vt3,3)=2

         curchgmrktab(k-1,1)=vertexlist(vt0,1)
         curchgmrktab(k-1+mxl*(ieor(up_pos,1)),1)=vertexlist(vt1,1)
      enddo
   elseif (crsign==-1) then
      !print*,crsign
      rmcounter=0d0
      stcounter=0d0
      do k=1,mxl
         s=bsites(k,b_pos)
         if (s==-1) cycle
         stcounter=stcounter+1
         up_pos=0
         leg_pos=k-1+mxl*up_pos
         vt0=mm_pos*dxl+leg_pos
         vt1=mm_pos*dxl+k-1+mxl*(ieor(up_pos,1))
         if (frststateop(s)==vt0 .and. vertexlist_map(vt1)==vt0) then
            frststateop(s)=-1
            vertexlist_map(vt0)=-1
            vertexlist_map(vt1)=-1
            vertexlist(vt0,3)=0
            vertexlist(vt1,3)=0
            custstateop(s)=-1
            !print*,"gen+=========================",loopnumber,loopnumper
            call loadlooptable(vertexlist(vt0,1))
            !print*,"gen00000000000000000000000000",loopnumber,loopnumper
            call loadlooptable(vertexlist(vt1,1))
            !print*,"gen--------------------------",loopnumber,loopnumper
            rmcounter=rmcounter+1
            loopnum0=loopnum0+1d0
            loopnumber=loopnumber+1d0

            vertexlist(vt0,:)=0d0
            vertexlist(vt1,:)=0d0
         else
            vt2=vertexlist_map(vt1)
            vt3=vertexlist_map(vt0)
            !vertexlist_map(vt0)=-1
            !vertexlist_map(vt1)=-1
            custstateop(s)=vt3
            if (frststateop(s)==vt0) then
               frststateop(s)=vt2
            endif
            !print*,"gen+=========================",loopnumber,loopnumper
            call loadlooptable(vertexlist(vt2,1))
            !print*,"gen00000000000000000000000000",loopnumber,loopnumper
            call loadlooptable(vertexlist(vt3,1))
            !print*,"gen--------------------------",loopnumber,loopnumper
            vertexlist_map(vt2)=vt3
            vertexlist_map(vt3)=vt2
            vertexlist(vt0,3)=0
            vertexlist(vt1,3)=0
            vertexlist(vt2,3)=2
            vertexlist(vt3,3)=2

            vertexlist(vt0,:)=0d0
            vertexlist(vt1,:)=0d0

            curchgmrktab(k-1,1)=vertexlist(vt3,1)
            curchgmrktab(k-1+mxl*(ieor(up_pos,1)),1)=vertexlist(vt2,1)
         endif
         !if (vertexlist(vt0,1)/=0d0 .or. vertexlist(vt0,2)/=0d0 &
         !   &.or. vertexlist(vt1,1)/=0d0 .or. vertexlist(vt1,2)/=0d0) then
         !   print*,"not clear"
         !   pause
         !endif
      enddo
   endif

   chgloopidx(:,:)=-1
   chgloopnum=0d0
   do k=0,dxl-1
      if (curchgmrktab(k,1)==-1 .or. curchgmrktab(k,2)/=0 .or. curchgmrktab(k,1)==0d0) then
         cycle
      else
         chgloopidx(chgloopnum,1)=curchgmrktab(k,1)
         chgloopnum=chgloopnum+1
         curchgmrktab(k,2)=curchgmrktab(k,2)+1
        do j=k,dxl
            if (curchgmrktab(j,1)==-1 .or. curchgmrktab(k,1)/=curchgmrktab(j,1)) then
               cycle
            else
               curchgmrktab(j,2)=curchgmrktab(j,2)+1
            endif
         enddo
      endif
   enddo

end subroutine gencurstateop
!==================================================!


!==================================================!
subroutine updateoporder(mm_pos,crsign)
   use configuration
   use measurementdata
   implicit none

   integer :: mm_pos,opod1,opod0,mm_0,mm_1,mm_tmp,opod_tmp1,opod_tmp2
   integer :: i,k,crsign,opod_pos,sz,tmp,ersign
   integer, allocatable :: opordercopy(:)
   !integer, allocatable :: loopoprecord_copy(:,:,:)
   i=size(oporder,dim=1)
   if (i<nh) then
      allocate(opordercopy(i))
      opordercopy(:)=oporder(:)
      deallocate(oporder)
      allocate(oporder(2*nh))
      oporder(:)=-1
      oporder(1:i)=opordercopy(1:i)
   endif
   call recordlooprelate(-2,-1,-1,-1)

   if (crsign==1) then
      if (nh==1) then
         tauscale(:)=1
         oporder(1)=mm_pos
         !loopoprecord(1,:,:)=-1
      else
         opod0=tauscale(mm_pos)
         opod1=mod(opod0,nh-1)+1
         mm_0=oporder(opod0)
         mm_1=oporder(opod1)

         if (mm_1==mm_0) then
            if (mm_pos<mm_0) then
               oporder(opod0+1:nh)=oporder(opod0:nh-1)
               !loopoprecord(opod0+1:nh,:,:)=loopoprecord(opod0:nh-1,:,:)
               opod_pos=opod0
               opod0=mod(opod0,nh)+1
            else
               opod_pos=nh
            endif
         elseif (mm_1<mm_0) then
            if (mm_pos>mm_0) then
               opod_pos=nh
            elseif (mm_pos<mm_1) then
               oporder(opod1+1:nh)=oporder(opod1:nh-1)
               !loopoprecord(opod0+1:nh,:,:)=loopoprecord(opod1:nh-1,:,:)
               opod_pos=opod1
               opod1=mod(opod1,nh)+1
            endif
         else
            oporder(opod1+1:nh)=oporder(opod1:nh-1)
            !loopoprecord(opod1+1:nh,:,:)=loopoprecord(opod1:nh-1,:,:)
            opod_pos=opod1
            opod1=mod(opod1,nh)+1
         endif
         oporder(opod_pos)=mm_pos
         !loopoprecord(opod_pos,:,:)=-1


         !print*,"0",opod0,mm_0,mm_pos,nh
         !print*,"1",opod1,mm_1,mm_pos,nh
         !print*,"sta",opod_pos,mm_pos
         !print*,"==============================="
         !pause

         !print*,"++++++++++++++++++++++++++++++++++++++++++"
         !sz=size(oporder,dim=1)
         !print*,"0",opod0,mm_0,mm_pos,nh
         !print*,"1",opod1,mm_1,mm_pos,nh
         !print*,"sta",opod_pos,mm_pos
         !print*,"==============================="
         !tmp=-1
         !ersign=0
         !do i=1,sz
         !  print*,"oporder",i,oporder(i),nh,size(oporder,dim=1)
         !  if (tmp>=oporder(i) .and. i<=nh) then
         !     ersign=1
         !     !pause
         !  endif
         !  tmp=oporder(i)
         !enddo

         if (opod_pos/=nh) then
            do i=opod_pos,nh-1
               opod_tmp1=i
               opod_tmp2=mod(opod_tmp1,nh)+1
               tauscale(oporder(opod_tmp1):oporder(opod_tmp2)-1)=opod_tmp1
            enddo
         endif
         tauscale(oporder(nh):mm-1)=nh
         tauscale(0:oporder(1)-1)=nh
      endif
   elseif (crsign==-1) then
      if (nh==0) then
         tauscale(:)=-1
         oporder(:)=-1
         !loopoprecord(:,:,:)=-1
      else
         opod_pos=tauscale(mm_pos)
         opod0=mod(opod_pos-2,nh+1)+1
         opod1=mod(opod_pos,nh+1)+1
         mm_0=oporder(opod0)
         mm_1=oporder(opod1)

         !print*,"]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]"
         !sz=size(oporder,dim=1)
         !print*,"0",opod0,mm_0,mm_pos,nh
         !print*,"1",opod1,mm_1,mm_pos,nh
         !print*,"sta",opod_pos,mm_pos
         !print*,"==============================="
         !tmp=-1
         !ersign=0
         !do i=1,sz
         !  print*,"oporder",i,oporder(i),nh,size(oporder,dim=1)
         !  if (tmp>=oporder(i) .and. i<=nh) then
         !     ersign=1
         !     !pause
         !  endif
         !  tmp=oporder(i)
         !enddo

         if (opod_pos==nh+1) then
            oporder(opod_pos)=-1
         else
            oporder(opod_pos:nh)=oporder(opod1:nh+1)
            oporder(nh+1)=-1
            !loopoprecord(opod_pos:nh,:,:)=loopoprecord(opod1:nh+1,:,:)
            !loopoprecord(nh+1,:,:)=-1
            !print*,"opod_pos:nh",opod_pos,nh
            !print*,"opod1:nh+1",opod1,nh+1
         endif

         !sz=size(oporder,dim=1)
         !print*,"0",opod0,mm_0,mm_pos,nh
         !print*,"1",opod1,mm_1,mm_pos,nh
         !print*,"sta",opod_pos,mm_pos
         !print*,"==============================="
         !tmp=-1
         !ersign=0
         !do i=1,sz
         !  print*,"oporder",i,oporder(i),nh,size(oporder,dim=1)
         !  if (tmp>=oporder(i) .and. i<=nh) then
         !     ersign=1
         !     !pause
         !  endif
         !  tmp=oporder(i)
         !enddo
         !if (ersign==1) pause

         if (opod0/=nh) then
            do i=opod0,nh-1
               opod_tmp1=i
               opod_tmp2=mod(opod_tmp1,nh)+1
               tauscale(oporder(opod_tmp1):oporder(opod_tmp2)-1)=opod_tmp1
            enddo
         endif
         tauscale(oporder(nh):mm-1)=nh
         tauscale(0:oporder(1)-1)=nh
      endif
   endif


end subroutine updateoporder
!==================================================!


!==================================================!
subroutine loadlooptable(i) 
   use configuration
   use measurementdata
   implicit none

   integer :: i,l,j,k,tmp,tmp2
   integer, allocatable :: rebootloopcopy(:)
   
   !print*,"idx",i
   !print*,"reboot",rebootnum,rebootloop(:)
   !print*,"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[["
   if (i/=0) then
      rebootnum=rebootnum+1
      l=size(rebootloop,dim=1)
      !print*,"load",size(rebootloop,dim=1),rebootnum,i
      if (l>=rebootnum) then
         tmp=i
         do j=1,rebootnum
            !print*,rebootloop(j),tmp,j,rebootnum,loopnumber
            if (rebootloop(j)==0) then
               rebootloop(j)=tmp
               loopnum1=loopnum1-1
               loopstate(i,1)=0d0
               loopnumber=loopnumber-1
               !print*,"loopnumber=loopnumber-1"
               !print*,tmp,loopnum1,sum(loopstate(:,1))
            elseif (rebootloop(j)>tmp) then
               tmp2=rebootloop(j)
               rebootloop(j)=tmp
               tmp=tmp2
            elseif (rebootloop(j)<tmp) then
            elseif (rebootloop(j)==tmp) then
               rebootnum=rebootnum-1
               exit
            endif
         enddo
      else
         allocate(rebootloopcopy(l))
         rebootloopcopy(:)=rebootloop(:)
         deallocate(rebootloop)
         allocate(rebootloop((2*l)))
         rebootloop(:)=0d0
         rebootloop(1:l)=rebootloopcopy(1:l)
         tmp=i
         do j=1,rebootnum
            !print*,rebootloop(j),tmp,j,rebootnum,loopnumber
            if (rebootloop(j)==0) then
               rebootloop(j)=tmp
               loopnum1=loopnum1-1
               loopnumber=loopnumber-1
               loopstate(i,1)=0d0
               !print*,tmp
               !print*,"loopnumber=loopnumber-1"
            elseif (rebootloop(j)>tmp) then
               tmp2=rebootloop(j)
               rebootloop(j)=tmp
               tmp=tmp2
            elseif (rebootloop(j)<tmp) then
            elseif (rebootloop(j)==tmp) then
               rebootnum=rebootnum-1
               exit
            endif
         enddo
      endif

      !print*,"loopnumber",loopnumber
      !print*,"reboot",rebootnum,rebootloop(:)
      !print*,i
      !if (l>nn) pause
   endif

end subroutine loadlooptable
!==================================================!
!==================================================!

subroutine turn(i,statechange)
!-----------------------------------------------------------!
!-----------------------------------------------------------!
 use configuration
 use vertexupdate
 implicit none

 integer :: i,legold,legnew,upold,upnew,opsmold,opsmnew,bstmp
 integer :: temp,statechange,stateupdate
 integer :: s1,s2
 temp=0d0

 opsmold=mod(opstring(i),2)
 legold=mod(v1,dxl)
 upold=int(legold/mxl)
 legold=mod(legold,mxl) 
 bstmp=int(legold/2)

 legnew=ieor(legold,1)
 upnew=upold
 v2=dxl*i+mod(upnew*mxl+legnew,dxl)
 s1=mod(v1,mxl)+1
 s2=mod(v2,mxl)+1
 !print*,s1,s2
 
 vertexlist(v1,1)=loopnumt
 vertexlist(v2,1)=loopnumt 
 vertexlist(v1,2)=looporder
 looporder=looporder+1
 vertexlist(v2,2)=looporder
 looporder=looporder+1
 vertexlist(v1,5)=loopphase
 loopphase=-loopphase
 vertexlist(v2,5)=loopphase
 !print*,"v1",v1,vertexlist(v1,2),"v2",v2,vertexlist(v2,2)
 call recordlooprelate(i,v1,v2,1)
 site2=site1
 site1=-1
 OpTy1=-1
 v1=vertexlist_map(v2)
 if (vertexlist_map(v1)/=v2) then
   print*,"map error",vertexlist_map(v1),v2,v1
   pause
 endif
 
updpath_counter=updpath_counter+1
 
 !if (Spm_measure_signal==1) print*,"turn",i

 end subroutine turn

!======================================================++++++++++++!

subroutine jump(i,statechange)
!-----------------------------------------------------------!
!-----------------------------------------------------------!
 use configuration
 use vertexupdate
 implicit none

 integer :: i,legold,legnew,upold,upnew,opsmold,opsmnew,bstmp
 integer :: temp,statechange,stateupdate,crsign
 integer :: s1,s2
 temp=0d0

 opsmold=mod(opstring(i),2)
 legold=mod(v1,dxl)
 upold=int(legold/mxl)
 legold=mod(legold,mxl) 
 bstmp=int(legold/2)

 legnew=ieor(legold,1)
 upnew=ieor(upold,1)
 v2=dxl*i+mod(upnew*mxl+legnew,dxl)
 s1=mod(v1,mxl)+1
 s2=mod(v2,mxl)+1
 !print*,s1,s2
 
 vertexlist(v1,1)=loopnumt
 vertexlist(v2,1)=loopnumt
 vertexlist(v1,2)=looporder
 looporder=looporder+1
 vertexlist(v2,2)=looporder
 looporder=looporder+1
 vertexlist(v1,5)=loopphase
 loopphase=loopphase
 vertexlist(v2,5)=loopphase
 !print*,"v1",v1,vertexlist(v1,2),"v2",v2,vertexlist(v2,2)
 call recordlooprelate(i,v1,v2,2)
 site2=site1
 site1=-1
 OpTy1=-1
 v1=vertexlist_map(v2)
 if (vertexlist_map(v1)/=v2) then
   print*,"map error",vertexlist_map(v1),v2,v1
   pause
 endif

 updpath_counter=updpath_counter+1
 
 !if (Spm_measure_signal==1) print*,"turn",i

 end subroutine jump
!======================================================++++++++++++!

subroutine strg(i,statechange)
!-----------------------------------------------------------!
!-----------------------------------------------------------!
 use configuration
 use vertexupdate
 implicit none

 integer :: i,legold,legnew,upold,upnew,opsmold,opsmnew,bstmp
 integer :: temp,statechange,stateupdate
 integer :: s1,s2
 temp=0d0

 opsmold=mod(opstring(i),2)
 legold=mod(v1,dxl)
 upold=int(legold/mxl)
 legold=mod(legold,mxl) 
 bstmp=int(legold/2)

 legnew=legold
 upnew=ieor(upold,1)
 v2=dxl*i+mod(upnew*mxl+legnew,dxl)
 s1=mod(v1,mxl)+1
 s2=mod(v2,mxl)+1
 !print*,s1,s2
 
 vertexlist(v1,1)=loopnumt
 vertexlist(v2,1)=loopnumt
 vertexlist(v1,2)=looporder
 looporder=looporder+1
 vertexlist(v2,2)=looporder
 looporder=looporder+1
 vertexlist(v1,5)=loopphase
 loopphase=loopphase
 vertexlist(v2,5)=loopphase
 !print*,"v1",v1,vertexlist(v1,2),"v2",v2,vertexlist(v2,2)
 call recordlooprelate(i,v1,v2,0)
 site2=site1
 site1=-1
 OpTy1=-1
 v1=vertexlist_map(v2)
 if (vertexlist_map(v1)/=v2) then
   print*,"map error",vertexlist_map(v1),v2,v1
   pause
 endif

 updpath_counter=updpath_counter+1
 
 !if (Spm_measure_signal==1) print*,"turn",i

 end subroutine strg
!==================================================!

subroutine adjustcutoff(step)
   use configuration
   implicit none

   integer, allocatable :: tauscalecopy(:)
   integer, allocatable :: stringcopy(:)
   integer, allocatable :: opdhasecopy(:)
   integer, allocatable :: vertexlistcopy(:,:)
   integer, allocatable :: vertexlist_mapcopy(:)
   integer :: mmnew,step

   mmnew=nh+nh/3
   !print*,"adj"
   if (mmnew<=mm) return

   allocate(stringcopy(0:mm-1))
   stringcopy(:)=opstring(:)
   deallocate(opstring)
   allocate(opstring(0:mmnew-1))
   opstring(:)=0
   !print*,"check0"

   allocate(tauscalecopy(0:mm-1))
   tauscalecopy(:)=tauscale(:)
   
   deallocate(tauscale)
   allocate(tauscale(0:mmnew-1))
   tauscale(0:mm-1)=tauscalecopy(:)
   tauscale(mm:mmnew-1)=nh
   !print*,"check1"

   allocate(opdhasecopy(0:mm-1))
   opdhasecopy(:)=opdhase(:)
   deallocate(opdhase)
   allocate(opdhase(0:mmnew-1))
   opdhase(:)=0d0
   !print*,"check2"

   opstring(0:mm-1)=stringcopy(:)
   opstring(mm:mmnew-1)=0
   deallocate(stringcopy)
   opdhase(0:mm-1)=opdhasecopy(:)
   opdhase(mm:mmnew-1)=0
   deallocate(opdhasecopy)
   !print*,"check3"

   allocate(vertexlistcopy(0:dxl*mm-1,5))
   vertexlistcopy(:,:)=vertexlist(:,:)
   deallocate(vertexlist)
   allocate(vertexlist(0:dxl*mmnew-1,5))
   vertexlist(:,:)=0d0
   vertexlist(0:dxl*mm-1,:)=vertexlistcopy(0:dxl*mm-1,:)
   deallocate(vertexlistcopy)
   !print*,"check4"

   allocate(vertexlist_mapcopy(0:dxl*mm-1))
   vertexlist_mapcopy(:)=vertexlist_map(:)
   deallocate(vertexlist_map)
   allocate(vertexlist_map(0:dxl*mmnew-1))
   vertexlist_map(:)=-1 
   vertexlist_map(0:dxl*mm-1)=vertexlist_mapcopy(0:dxl*mm-1)
   deallocate(vertexlist_mapcopy)
   !print*,"check5"

   mm=mmnew

   open(unit=10,file='info.dat',position='append')
   write(10,*)' Step: ',step,'  Cut-off L: ',mm,'  Current nh: ',nh
   close(10)
   !print*,"check6"

 end subroutine adjustcutoff
!==================================================!
!==================================================!
subroutine measure()
   use configuration
   use measurementdata
   implicit none

   integer :: i,b,op,s0,s1,s2,rx,ry,rz,x,y,z,dv,k,sig,j,st,l,stiff_cou,vt0,vt1,cphase,cstate
   real(8) :: am1,am2,am3,am4,dm1,dm2,dm1t,dm2t,amt2,amt4,am8,am9,amt8,amt9,dm6,tg,tm
   real(8) :: amt,apt,sp0,sp1,dmt1,dmt2,dmp1,dmp2,tmp,check,fact,dm3,dm4,dm5,dt1,dt2,dtp1,dtp2
   integer :: st0,st1,mrk0,mrk1,mrk2,looptmp,phs1,phs2,lp1,lp2,idxnum,istate,idx,vtmp
   integer :: s,s3,s4,x1,y1,z1
   real(8) :: sn1,sn2,foth_factor,stiff_factor
   real(8), allocatable :: windrecord(:,:)
   integer, allocatable :: looprecord(:,:)
   real(8), allocatable :: jj(:)
   real(8), allocatable :: ag(:)
   real(8), allocatable :: ss(:)
   real(8), allocatable :: tptm(:,:)
   real(8), allocatable :: saft(:)
   integer, allocatable :: am(:)
   real(8), external :: ran
   allocate(jj(2))
   allocate(ag(2))
   allocate(ss(3))
   allocate(am(2))
   allocate(saft(2))
   check=0d0
   am(:)=0d0
   saft(:)=0d0
   !print*,"=================================="

   looptmp=loopnummax

   allocate(looprecord(looptmp+nn,0:3))
   allocate(windrecord(looptmp,2))

   looprecord(:,:)=0d0
   windrecord(:,:)=0d0
   stiff_cou=0d0

   do i=1,nn
      if (custstateop(i)>=0) then
         cstate=loopstate(vertexlist(custstateop(i),1),0)
         cphase=mod(vertexlist(custstateop(i),5),2)
      else
         cstate=state(i,1)
         cphase=1
      endif
      spin(i)=cphase*cstate
   enddo
   am1=0.d0
   am2=am(1)

   ag(:)=0d0
   jj(:)=0d0
   ss(:)=0d0

   do i = 0, mm-1
      op=opstring(i)
      saft=0d0
      if (op/=0) then
         b=op/4
         call gencurstateop(i,b,0)
         do j=mxl,dxl-1
            vtmp=dxl*i+j
            cstate=loopstate(vertexlist(vtmp,1),0)
            cphase=vertexlist(vtmp,5)
            if (state(bsites(j+1-mxl,b),2)==0d0) then
               spin(bsites(j+1-mxl,b))=cstate*cphase
            endif
         enddo
      endif
      !print*,"check"

!===================================================================================++!

   !======================================================================+++!

      do l = 1, nn
         am(1)=am(1)+spin(l)
      enddo 

      !am1=am1+dfloat(abs(am))
      !am2=am2+dfloat(am)**2
      am1=am1+dfloat(am(1))
      !am2=am2+dfloat(abs(am(2)))

   end do

    !print*,"op off",opoff,opp,opm,opp-opm,ss(1)
    
   if (mm/=0) then
      am1=am1/dble(mm)
   else
      am1=dfloat(am(1))
   endif
   !if (jp/=0) am2=nh1o/dble(jp*beta)

   !print*,am1,am(1),nn

   tm=0d0
   do x1=0,lx-1
      do y1=0,ly-1
         do z1=0,lz-1
            s=1+x1+y1*lx+z1*lx*ly
            s1=1+mod(x1,lx)+mod(y1,ly)*lx+mod(z1,lz)*lx*ly
            s2=1+mod(x1,lx)+mod(y1,ly)*lx+mod(z1+1,lz)*lx*ly
            s3=1+mod(x1-1,lx)+mod(y1,ly)*lx+mod(z1+1,lz)*lx*ly
            s4=1+mod(x1,lx)+mod(y1-1,ly)*lx+mod(z1+1,lz)*lx*ly
            tg=0d0
            tg=spin(s1+0*sitnum)+spin(s1+1*sitnum)+spin(s1+2*sitnum)+spin(s1+3*sitnum)
            tm=tm+abs(tg)
            tg=0d0
            tg=spin(s2+0*sitnum)+spin(s3+1*sitnum)+spin(s4+2*sitnum)+spin(s1+3*sitnum)
            tm=tm+abs(tg)
         enddo
      enddo
   enddo
   
   !print*,tm
   !print*,ss(1),ss(2),ss(3)
   stiff(1)=stiff(1)+(ss(1))**2
   stiff(2)=stiff(2)+(ss(2))**2
   stiff(3)=stiff(3)+(ss(3))**2
   stiff(0)=stiff(0)+ss(1)**2+ss(2)**2+ss(3)**2

   amag1=amag1+am1
   amag2=amag2+am2
   bind1(1)=bind1(1)+(am1/dble(nn))**2
   bind2(1)=bind2(1)+(am2/dble(nn))**2
   bind1(2)=bind1(2)+(am1/dble(nn))**4
   bind2(2)=bind2(2)+(am2/dble(nn))**4
   tmag1=tmag1+tm

   !if ( check==0 ) print*,"warning",ss(:) 

   do s1=1, sitnum
      x=mod(s1-1,lx)
      y=mod(int((s1-1)/lx),ly)
      z=int((s1-1)/(lx*ly))
      do rx=0, lx-1
         do ry=0, ly-1
            do rz=0, lz-1
               s2=mod(x+rx,lx)+mod(y+ry,ly)*lx+mod(z+rz,lz)*lx*ly+1
               do i=1, 4
                  crr(rx,ry,rz,i)=crr(rx,ry,rz,i)+spin(s1+(i-1)*sitnum)*spin(s2+(i-1)*sitnum)
                  !if (rx/=0 .and. ry/=0 .and. rz/=0) then
                  !if ((s1+(i-1)*sitnum)==(s2+(i-1)*sitnum)) print*, (s1+(i-1)*sitnum),(s2+(i-1)*sitnum)
                  !endif
                  !print*, (s1+(i-1)*sitnum),(s2+(i-1)*sitnum)
               enddo
            enddo
         enddo
      enddo
   enddo

   enrg1=enrg1+dble(nh)
   enrg2=enrg2+dble(nh)**2
   amag1=amag1+am1
   nms1=nms1+1
   !print*,jj(1),jj(2),0.5d0*(dble(jj(1))+dble(jj(2)))

   !print*,"check_fin1",nh
   !print*,"check_fin1",nh
   deallocate(looprecord)
   !print*,"check_fin1",nh
   deallocate(windrecord)
   !print*,"check_fin2",nh
end subroutine measure
!==================================================!
!==================================================!

subroutine writeresult(rank)
   use configuration
   use measurementdata
   use vertexupdate
   implicit none

   integer :: msteps,rx,ry,rz,q,t,i1,i2,i3,rank,i,test
   real(8) :: dmt1,dmt2

   real(8), external :: qx
   real(8), external :: qy

    resname='res000.dat'
    odpname='odp000.dat'
    crrname='crr000.dat'
    tcorname='tcor000.dat'
    tcpmname='tcpm000.dat'

    i3=rank/100
    i2=mod(rank,100)/10
    i1=mod(rank,10)

    resname(6:6)=achar(48+i1)
    resname(5:5)=achar(48+i2)
    resname(4:4)=achar(48+i3)

    odpname(6:6)=achar(48+i1)
    odpname(5:5)=achar(48+i2)
    odpname(4:4)=achar(48+i3)

    crrname(6:6)=achar(48+i1)
    crrname(5:5)=achar(48+i2)
    crrname(4:4)=achar(48+i3)

    tcorname(7:7)=achar(48+i1)
    tcorname(6:6)=achar(48+i2)
    tcorname(5:5)=achar(48+i3)

    tcpmname(7:7)=achar(48+i1)
    tcpmname(6:6)=achar(48+i2)
    tcpmname(5:5)=achar(48+i3)


   enrg1=enrg1/dble(nms1)
   enrg2=enrg2/dble(nms1)
   amag1=amag1/dble(nms1)
   amag2=amag2/dble(nms1)
   tmag1=tmag1/dble(nms1)
   tmag1=tmag1/dble(4*2*sitnum)

   enrg2=(enrg2-enrg1*(enrg1+1.d0))/nn
   enrg1=-(enrg1/(beta*dble(nn)))
   if (nnn==1d0) then
      enrg1=enrg1+(pbAc*(addele)+pbBs*jp*(1d0/dble(2**6))+pbCn*jp*(1d0/dble(3**3)))/dble(nn)
   else
      enrg1=enrg1+(pbAc*(addele))/dble(nn)
   endif

   amag1=amag1/dble(nn)
   amag2=amag2/dble(nn)
   stiff=stiff/(beta*nn)
   
   open(10,file=resname,position='append')
   write(10,*)beta,hz,jp,enrg1,enrg2,amag1,amag2,tmag1
   close(10)

   open(10,file=odpname,position='append')
   write(10,*)amag2,amag4,dimr1,dimr2,dimr3
   close(10)

   enrg1=0.d0
   enrg2=0.d0
   amag1=0.d0
   amag3=0.d0
   stiff=0.d0

   crr=crr/dble(sitnum)/dble(nms1)
   open(20,file=crrname,position='append')
   do rx=0,lx-1
      do ry=0,ly-1
         do rz=0,lz-1
            write(20,*)rx,ry,rz,beta,hz,jp,crr(rx,ry,rz,:)
         enddo
      enddo
   end do
   close(20)
   crr=crr*dble(sitnum)*dble(nms1)

   nms1=0

   tcor=tcor/dble(dmweight)/dble(sun**2-1) 
   open(10,file=tcorname,position='append')
   do q=1,nn
      write(10,'(2i8)')q
      do t=0,ntau
            write(10,'(f20.12)')tcor(t,q)
      enddo
   enddo
   close(10)
   tcor=0.d0

   tcordm=tcordm/dble(dmweight)/dble(sun**2-1)
   tcordz=tcordz/dble(dmweight)/dble(sun-1)
   open(10,file=tcpmname,position='append')
   do q=1,nn
      write(10,'(2i8)')q
      do t=0,ntau
            write(10,'(f20.12)')tcordm(t,q,1)  
      enddo
   enddo
   close(10)

   tcordm(:,:,:)=0d0
   tcordz(:,:,:)=0d0
   nms2=0d0

end subroutine writeresult
!==================================================!



!==================================================!

real(8) function ran()
!----------------------------------------------!
! 64-bit congruental generator                 !
! iran64=oran64*2862933555777941757+1013904243 !
!----------------------------------------------!
 implicit none

 real(8) :: dmu64
 integer(8) :: ran64,mul64,add64
 common/bran64/dmu64,ran64,mul64,add64

 ran64=ran64*mul64+add64
 ran=0.5d0+dmu64*dble(ran64)

 end function ran


Subroutine initran(w,rank)
!--------------------------------------------------!
 implicit none
 integer(8) :: irmax
 integer(4) :: w,b
 integer :: rank,system_time
 real(8) :: dmu64
 integer(8) :: ran64,mul64,add64
 common/bran64/dmu64,ran64,mul64,add64
 
irmax=2_8**31
irmax=2*(irmax**2-1)+1
mul64=2862933555777941757_8
add64=1013904243
dmu64=0.5d0/dble(irmax)
call system_clock(system_time)
ran64=abs(system_time-(rank*1993+2018)*20160514)  

end Subroutine initran
!==================================================!

!==================================================!

subroutine deallocateall()
   use configuration
   use measurementdata
   implicit none

   deallocate(state)
   !print*,"state"
   deallocate(bsites)
   !print*,"bsites"
   deallocate(opstring)
   !print*,"opstring"
   deallocate(opdhase)
   !print*,"opdhase"
   deallocate(tauscale)
   !print*,"tauscale"
   deallocate(frststateop)
   !print*,"frststateop"
   deallocate(laststateop)
   !print*,"laststateop"
   deallocate(vertexlist)
   !print*,"vertexlist"
   deallocate(vertexlist_map)
   !print*,"vertexlist_map"
   deallocate(rantau)
   !print*,"rantau"
   deallocate(tc)
   !print*,"tc"
   deallocate(phi)
   !print*,"phi"
   deallocate(tcor)
   !print*,"tcor"
   deallocate(ref)
   !print*,"ref"
   deallocate(imf)
   !print*,"imf"
   deallocate(crr)
   !print*,"crr"

end subroutine deallocateall
!==================================================!        

!==================================================!     

subroutine taugrid(gtype,tmax)
!-------------------------------------------------------------------------!
!-constructs array tgrd(0,...,ntau) of time separations (in units of dtau)!
!-gtype = 1 for uniform grid of all separatiins up to tmax
!-        2 for quadratic grid t=dtau*n^2/4 up to beta/2
!-        3 for uniform grid up to tmax followed by quadratic to beta/2
!-actual time points including dtau factor are written to 'tgrid.dat'
!--------------------------------------------------------------------------!
 use configuration
 use measurementdata
 implicit none

 integer :: i,t1,t2,gtype,nsm
 real(8) :: tmax

 nsm=int((tmax+1.d-5)/dtau)
 open(10,file='tgrid.dat',status='replace')

 if (gtype==1) then
    ntau=nsm
    write(10,'(i4)')ntau
    allocate(tgrd(0:ntau)) 
    do i=0,ntau
       tgrd(i)=i
       write(10,'(f15.8)')tgrd(i)*dtau
    enddo

 elseif (gtype==2) then    
    ntau=0
    t1=0
    do
       t2=((ntau+1)**2)/4
       if (t2==t1) t2=t1+1
       if (t2.le.nsm) then
          ntau=ntau+1
          t1=t2
       else
          exit
       endif
    enddo
    write(10,'(i4)')ntau
    allocate(tgrd(0:ntau)) 
    tgrd(0)=0
    write(10,'(f15.8)')0.d0
    do i=1,ntau
       tgrd(i)=(i**2)/4
       if (tgrd(i)==tgrd(i-1)) tgrd(i)=tgrd(i-1)+1       
       write(10,'(f15.8)')tgrd(i)*dtau
    enddo

 elseif (gtype==3) then  
    ntau=nsm
    t1=0
    i=0
    do
       t2=((i+1)**2)/4
       if (t2==t1) t2=t1+1
       if (t2.gt.nsm.and.t2.le.ns/2) then
          ntau=ntau+1
          t1=t2
       elseif (t2.gt.ns/2) then
          exit
       endif
       i=i+1
    enddo
    write(10,'(i4)')ntau
    allocate(tgrd(0:ntau)) 
    tgrd(0)=0
    do i=0,nsm
       tgrd(i)=i
       write(10,'(f15.8)')tgrd(i)*dtau
    enddo
    ntau=nsm
    t1=0
    i=0
    do
       t2=((i+1)**2)/4
       if (t2==t1) t2=t1+1
       if (t2.gt.nsm.and.t2.le.ns/2) then
          ntau=ntau+1
          tgrd(ntau)=t2
          write(10,'(f15.8)')tgrd(ntau)*dtau
          t1=t2
       elseif (t2.gt.ns/2) then
          exit
       endif
       i=i+1
    enddo
 endif 
 close(10)

 allocate(tc(0:ntau)) 
 allocate(tcor_real(0:ntau,nn)) 
 allocate(tcor(0:ntau,nn)) 
 allocate(tcordm(0:ntau,nn,4))
 allocate(tcordz(0:ntau,nn,4))
 allocate(PMinRealSpace(nn,0:ntau))
 allocate(PMRS_temp(nn,0:ntau))
 allocate(DZinRealSpace(nn,0:ntau,4))
 allocate(DCinRealSpace(nn,0:ntau,4))
 allocate(demo_pm(nn,0:ntau))
 allocate(demo_or(nn,0:ntau))
 tcor_real=0d0
 tcor=0.d0
 tcordm=0d0
 tcordz=0d0
 nms3=0d0
 PMRS_temp=0d0
 PMinRealSpace=0d0
 demo_pm=0d0
 demo_or=0d0
 nc=0d0
 DZinRealSpace=0d0
 DCinRealSpace=0d0
 dmweight=0d0
 stweight=0d0

 end subroutine taugrid
 !==================================================!    



!==================================================!

 subroutine random_tau()
!---------------------------------------------!
!time dependent q-space correlation functions S+S-
!---------------------------------------------!
 use configuration
 use measurementdata
 implicit none

 integer :: i,j,k
 real(8), external :: ran

 if ( size(rantau, dim=1)/=nh+1 ) then
   deallocate(rantau)
   allocate(rantau(nh))
 endif 

 rantau(nh)=beta
 do i=1,nh
   rantau(i)=ran()*beta
 enddo

 call quick_sort(rantau,nh,1,nh)
 
 end subroutine random_tau
!==================================================!
!==================================================!

recursive subroutine quick_sort(a,n,s,e)
implicit none
integer :: n      
real(8) :: a(1:n+1) 
integer :: s      
integer :: e      
integer :: l,r    
real(8) :: k      
real(8) :: temp   
l=s
r=e+1
if ( r<=l ) return

k=a(s) 
do while(.true.)

   do while( .true. )
      l=l+1
      if ( (a(l) > k) .or. (l>=e) ) exit
   end do

   do while( .true. )
      r=r-1
      if ( (a(r) < k) .or. (r<=s) ) exit
   end do
   if ( r <= l ) exit

   temp=a(l)
   a(l)=a(r)
   a(r)=temp
end do
 
temp=a(s)
a(s)=a(r)
a(r)=temp

call quick_sort(a,n,s,r-1) 
call quick_sort(a,n,r+1,e)
return
end subroutine quick_sort
!==================================================!

!==================================================!

recursive subroutine quick_sort_int(a,n,s,e)
implicit none
integer :: n      
integer :: a(1:n+1) 
integer :: s      
integer :: e      
integer :: l,r    
integer :: k      
integer :: temp   
l=s
r=e+1
if ( r<=l ) return

k=a(s) 
do while(.true.)

   do while( .true. )
      l=l+1
      if ( (a(l) > k) .or. (l>=e) ) exit
   end do

   do while( .true. )
      r=r-1
      if ( (a(r) < k) .or. (r<=s) ) exit
   end do
   if ( r <= l ) exit

   temp=a(l)
   a(l)=a(r)
   a(r)=temp
end do
 
temp=a(s)
a(s)=a(r)
a(r)=temp

call quick_sort_int(a,n,s,r-1) 
call quick_sort_int(a,n,r+1,e)
return
end subroutine quick_sort_int
!==================================================!
   
subroutine qarrays()
!-----------------------------------------------------------!
! phase factors for fourier transforms of state configuration
!-----------------------------------------------------------!
 use configuration
 use measurementdata
 implicit none

 integer :: r,q
 real(8), external :: qx
 real(8), external :: qy
 real(8) :: qqx,qqy

 allocate(phi(2,nn,nq_all))
 allocate(ref(ns,nq_all))
 allocate(imf(ns,nq_all))

 if (ly>1) then
   do q=1, nq_all
      qqx=2.d0*pi*dble(qx(q-1))/dble(lx)
      qqy=2.d0*pi*dble(qy(q-1))/dble(ly)
      do r=1,nn
            phi(1,r,q)=cos(dble(mod((r-1),lx))*qqx+dble((r-1)/lx)*qqy)/sqrt(dble(nn))       
            phi(2,r,q)=sin(dble(mod((r-1),lx))*qqx+dble((r-1)/lx)*qqy)/sqrt(dble(nn)) 
      enddo
   enddo
  else
   do q=1, nq_all
      qqx=2.d0*pi*dble(q-1)/dble(lx)
      qqy=2.d0*pi*dble(0)/dble(ly)
      do r=1,nn
            phi(1,r,q)=cos(dble(mod((r-1),lx))*qqx+dble((r-1)/lx)*qqy)/sqrt(dble(nn))       
            phi(2,r,q)=sin(dble(mod((r-1),lx))*qqx+dble((r-1)/lx)*qqy)/sqrt(dble(nn)) 
      enddo
   enddo
  endif


 end subroutine qarrays
!----------------------!
!==================================================!
   
subroutine initweight()
!-----------------------------------------------------------!
! phase factors for fourier transforms of state configuration
!-----------------------------------------------------------!
 use configuration
 use measurementdata
 implicit none

 integer :: i,j,k,s1,s2,p1,p2
 addele=(Ac*0.25d0+hz/3d0*0.5d0)+0.25d0
 print*,"addele",addele

 weight(:,:)=0d0
 fwight(:,:)=0d0
 vex2weight(:,:,:,:)=0d0

 weight(1,0)=(addele-Ac*1d0*0.25d0-(1d0+1d0)*hz/6d0*0.5d0)!0 for - 1 for +
 if (weight(1,0)/=0d0) weight(4,0)=weight(4,0)+1
 weight(1,1)=(addele+Ac*1d0*0.25d0-(1d0-1d0)*hz/6d0*0.5d0)!0 for - 1 for +
 if (weight(1,1)/=0d0) weight(4,1)=weight(4,1)+1
 weight(1,2)=(addele-Ac*1d0*0.25d0+(1d0+1d0)*hz/6d0*0.5d0)!0 for - 1 for +
 if (weight(1,2)/=0d0) weight(4,2)=weight(4,2)+1
 weight(1,3)=0
 if (weight(1,3)/=0d0) weight(4,3)=weight(4,3)+1
 weight(1,4)=(weight(1,4)-1)*sum(weight(1,0:3))
 !print*,"w1",weight(1,:)

 weight(2,0)=0d0!0 for - 1 for +
 if (weight(2,0)/=0d0) weight(4,0)=weight(4,0)+1
 weight(2,1)=(addele+Ac*1d0*0.25d0-(1d0-1d0)*hz/6d0*0.5d0)!0 for - 1 for +
 if (weight(2,1)/=0d0) weight(4,1)=weight(4,1)+1
 weight(2,2)=0d0!0 for - 1 for +
 if (weight(2,2)/=0d0) weight(4,2)=weight(4,2)+1
 weight(2,3)=jp
 if (weight(2,3)/=0d0) weight(4,3)=weight(4,3)+1
 weight(2,4)=(weight(2,4)-1)*sum(weight(2,0:3))
 !print*,"w2",weight(2,:)

 weight(3,0)=(addele-Ac*1d0*0.25d0-(1d0+1d0)*hz/6d0*0.5d0)!0 for - 1 for +
 if (weight(3,0)/=0d0) weight(4,0)=weight(4,0)+1
 weight(3,1)=0d0
 if (weight(3,1)/=0d0) weight(4,1)=weight(4,1)+1
 weight(3,2)=(addele-Ac*1d0*0.25d0+(1d0+1d0)*hz/6d0*0.5d0)!0 for - 1 for +
 if (weight(3,2)/=0d0) weight(4,2)=weight(4,2)+1
 weight(3,3)=jp
 if (weight(3,3)/=0d0) weight(4,3)=weight(4,3)+1
 weight(3,4)=(weight(3,4)-1)*sum(weight(3,0:3))
 !print*,"w3",weight(3,:)

 do i=1,3
    fwight(i,0)=min(weight(i,0),weight(i,2))
    fwight(i,1)=weight(i,1)
    fwight(i,2)=min(weight(i,0),weight(i,2))
    fwight(i,3)=weight(i,3)
    !print*,"fwight",fwight(i,0:3)
    !print*,"weight",weight(i,0:3)
 enddo

 print*,"cn",weight(4,:)

 change(:,:)=0d0
 do i=0,3
   do j=0,3
      if (weight(i,j)/=0d0) then
         do k=0,3
            if (k/=j) then
               !change(i,j)=change(i,j)+weight(i,k)
               if (weight(i,k)/=0d0) then
                  change(i,j)=1d0
               endif
            endif
         enddo
      else
         change(i,j)=0
      endif
   enddo
   change(i,4)=sum(change(i,0:3))
 enddo

 do i=0,3
   !do j=0,3
   !  if (change(i,j)/=0d0) then
   !     change(4,j)=change(4,j)+1
   !  endif
   !enddo
   change(4,i)=sum(change(0:3,i))
 enddo

 do i=1,4
   print*,"change",change(i,:)
 enddo
 !pause

 vex2weight(:,:,:,:)=0d0
 do i=1,12
   if (mod(i-1,4)==0) then
      p1=1
      p2=1
   elseif (mod(i-1,4)==1) then
      p1=-1
      p2=-1
   elseif (mod(i-1,4)==2) then
      p1=1
      p2=-1
   elseif (mod(i-1,4)==3) then
      p1=-1
      p2=1
   endif
   do j=0,1
      s1=(j*2-1)
      do k=0,1
         s2=(k*2-1)
            if (int((i-1)/4)/=1) then
               vex2weight(i,j,k,1)=int((i-1)/4)+1
               vex2weight(i,j,k,2)=int((s1*p1+s2*p2+2)/2d0)
            else
               vex2weight(i,j,k,1)=int((i-1)/4)+1
               vex2weight(i,j,k,2)=int((0+2)/2d0)           
            endif
      enddo
   enddo
 enddo
 vex2weight(5:6,1,0,2)=3
 vex2weight(5:6,0,1,2)=3
 vex2weight(7:8,1,1,2)=3
 vex2weight(7:8,0,0,2)=3

 vex2weight(9:10,1,0,2)=3
 vex2weight(9:10,0,1,2)=3
 vex2weight(11:12,1,1,2)=3
 vex2weight(11:12,0,0,2)=3

 end subroutine initweight
!==================================================!
   
subroutine phase2vex(ltyp_cur,idx_0,idx_1)
!-----------------------------------------------------------!
! phase factors for fourier transforms of state configuration
!-----------------------------------------------------------!
 use configuration
 use measurementdata
 implicit none

 integer :: i,j,k,ltyp_cur,idx_0,idx_1

 if (ltyp_cur==0d0) then
   if (phase2jdg(0,2)==phase2jdg(2,2) .and. phase2jdg(1,2)==phase2jdg(3,2)) then
      if (phase2jdg(0,2)==phase2jdg(1,2)) then !==================================================1 2
         if ((phase2jdg(0,1)==idx_0 .or. phase2jdg(1,1)==idx_0) .and. phase2jdg(0,2)==1) then
            vex2record=1
         elseif ((phase2jdg(0,1)==idx_0 .or. phase2jdg(1,1)==idx_0) .and. phase2jdg(0,2)==-1) then
            vex2record=2
         endif
      elseif (phase2jdg(0,2)==-phase2jdg(1,2)) then!==================================================3 4
         if (phase2jdg(0,1)==idx_0) then
            if (phase2jdg(0,2)==1) then
               vex2record=3
            elseif (phase2jdg(0,2)==-1) then
               vex2record=4
            endif
         elseif (phase2jdg(1,1)==idx_0) then
            if (phase2jdg(1,2)==1) then
               vex2record=3
            elseif (phase2jdg(1,2)==-1) then
               vex2record=4
            endif
         endif
      endif
   endif
 elseif (ltyp_cur==1d0) then
   if (phase2jdg(0,2)==-phase2jdg(1,2) .and. phase2jdg(2,2)==-phase2jdg(3,2)) then 
      if (phase2jdg(0,2)==phase2jdg(2,2)) then!=======================================================5 6
         if ((phase2jdg(0,1)==idx_0 .or. phase2jdg(2,1)==idx_0) .and. phase2jdg(0,2)==1) then
            vex2record=5
         elseif ((phase2jdg(0,1)==idx_0 .or. phase2jdg(2,1)==idx_0) .and. phase2jdg(0,2)==-1) then
            vex2record=6
         endif
      elseif (phase2jdg(0,2)==-phase2jdg(2,2)) then!==================================================7 8
         if (phase2jdg(0,1)==idx_0) then
            if (phase2jdg(0,2)==1) then
               vex2record=7
            elseif (phase2jdg(0,2)==-1) then
               vex2record=8
            endif
         elseif (phase2jdg(2,1)==idx_0) then
            if (phase2jdg(2,2)==1) then
               vex2record=7
            elseif (phase2jdg(2,2)==-1) then
               vex2record=8
            endif
         endif
      endif
   endif
 elseif (ltyp_cur==2d0) then
   if (phase2jdg(0,2)==phase2jdg(3,2) .and. phase2jdg(1,2)==phase2jdg(2,2)) then
      if (phase2jdg(0,2)==phase2jdg(2,2)) then!==================================================9 10
         if ((phase2jdg(0,1)==idx_0 .or. phase2jdg(1,1)==idx_0) .and. phase2jdg(0,2)==1) then
            vex2record=9
         elseif ((phase2jdg(0,1)==idx_0 .or. phase2jdg(1,1)==idx_0) .and. phase2jdg(0,2)==-1) then
            vex2record=10
         endif
      elseif (phase2jdg(0,2)==-phase2jdg(2,2)) then!==================================================11 12
         if (phase2jdg(0,1)==idx_0) then
            if (phase2jdg(0,2)==1) then
               vex2record=11
            elseif (phase2jdg(0,2)==-1) then
               vex2record=12
            endif
         elseif (phase2jdg(2,1)==idx_0) then
            if (phase2jdg(2,2)==1) then
               vex2record=11
            elseif (phase2jdg(2,2)==-1) then
               vex2record=12
            endif
         endif
      endif
   endif
 endif

 end subroutine phase2vex
!----------------------!


real(8) function qx(q)
!----------------------------------------------!
!-find qx point-!
!----------------------------------------------!
 use configuration
 use measurementdata

 implicit none

 integer :: q

 if ( int(q/nq) == 0 ) then
   qx=mod(q,nq)
 elseif(int(q/nq) == 1) then
   qx=nq 
 elseif(int(q/nq) == 2) then
   qx=nq-mod(q,nq)
 elseif(int(q/nq) == 3) then
   qx=0
 elseif(int(q/nq) == 4) then
   qx=mod(q,nq)
 elseif(int(q/nq) == 5) then
   qx=nq-mod(q,nq)
 elseif(int(q/nq) == 6) then
   qx=0
 endif

 end function qx

 !----------------------------------------------!
 !----------------------------------------------!
 !----------------------------------------------!

 real(8) function qy(q)
!----------------------------------------------!
!-find qy point-!
!----------------------------------------------!
 use configuration
 use measurementdata

 implicit none

 integer :: q

 if (int(q/nq) == 0) then
   qy=mod(q,nq)
 elseif(int(q/nq) == 1) then
   qy=nq-mod(q,nq)
 elseif(int(q/nq) == 2) then
   qy=0
 elseif(int(q/nq) == 3) then
   qy=mod(q,nq)
 elseif(int(q/nq) == 4) then
   qy=nq
 elseif(int(q/nq) == 5) then
   qy=nq-mod(q,nq)
 elseif(int(q/nq) == 6) then
   qy=0
 endif

 end function qy

!----------------------------------------------!
!----------------------------------------------!

subroutine writeconfig(rank)
!--------------------------------------!
 use configuration 
 implicit none
 integer :: i,rank
 character(len=3) :: ranks
 write (ranks,fmt='(i3.3)') rank
 open(10,file='conf'//ranks//'.log',status='replace')
 do i=1,nn
    write(10,'(i2)')state(i,:)
 enddo
 write(10,*)mm,nh
 do i=0,mm-1
    write(10,*)opstring(i)
 enddo
 do i=0,dxl*mm-1
    write(10,*)vertexlist_map(i),vertexlist(i,:)
 enddo
 write(10,*)loopnum0,loopnum1,loopnumber
 close(10)
 end subroutine writeconfig
!------------------------!

 subroutine readconfig(rank)
!---------------------!
 use configuration
 implicit none
 integer :: i,rank,j,op
 character(len=3) :: ranks
 write (ranks,fmt='(i3.3)') rank
 print*, 'conf'//ranks//'.log'
 open(10,file='conf'//ranks//'.log',status='old')
 do i=1,nn
    read(10,*)state(i,:)
 enddo
 read(10,*)mm,nh
 allocate(opstring(0:mm-1))
 allocate(opdhase(0:mm-1))
 allocate(tauscale(0:mm-1))
 allocate(vertexlist(0:dxl*mm-1,5))
 allocate(vertexlist_map(0:dxl*mm-1))
 allocate(rantau(nh)) 
 allocate(oporder(2*nh))
 do i=0,mm-1
    read(10,*)opstring(i)
 enddo
 tauscale(:)=0
 do i=0,dxl*mm-1
    read(10,*)vertexlist_map(i),vertexlist(i,:)
 enddo
 read(10,*)loopnum0,loopnum1,loopnumber
 close(10)

 j=nh
 do i=0,mm-1
      op=opstring(i)
      if (op==0) then
         tauscale(i)=j
         cycle
      endif
    j=mod(j,nh)+1
    oporder(j)=i
    tauscale(i)=j
 enddo
 end subroutine readconfig

!----------------------------------------------!
!----------------------!
