"""
Baseline models for AUV dynamics on SE(3).

These models are organized as structural comparisons against ``AUVHamNODE``:

  - ``AUVHamNODEMergedNC``:
      keep mass + scalar potential + co-adjoint + exact SE(3), but merge
      D/J/B into one learned non-conservative force
  - ``AUVHamNODEQForce``:
      keep mass + co-adjoint + exact SE(3) + separate D/J/B, but replace the
      conservative force induced by a scalar potential V(q) with a generic
      configuration-dependent generalized force g(q)
  - ``AUVMomentumSE3NODE``:
      keep mass / momentum coordinates + exact SE(3), remove co-adjoint and
      explicit conservative/non-conservative decomposition
  - ``AUVUnstructuredHamNODE``:
      keep SE(3) Hamiltonian equations with a single learned H(q, p)
  - ``AUVUnstructuredSE3NODE``:
      keep exact SE(3) kinematics only
  - ``AUVBlackBoxNODE``:
      learn the full state derivative, including kinematics

All models share:
  - the augmented solver state used by ``AUVHamNODE``
  - first-order actuator lag with learnable time constants
  - the same ocean-current state convention:
      dataset stores total body velocity nu, ODE state stores relative
      body velocity nu_r when ``ocean_current=True``
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Three-layer perceptron with orthogonal initialization."""

    def __init__(self, in_dim, hidden, out_dim, activation='tanh', gain=0.1):
        super().__init__()
        act = {'tanh': nn.Tanh, 'softplus': nn.Softplus, 'relu': nn.ReLU}[activation]()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), act,
            nn.Linear(hidden, hidden), act,
            nn.Linear(hidden, out_dim),
        )
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)


@dataclass(frozen=True)
class StateLayout:
    """State layout for the augmented solver state."""

    pose_dim: int
    nu_dim: int
    u_dim: int
    ocean_current: bool
    absolute_depth_context: bool

    @property
    def pos(self) -> slice:
        return slice(0, 3)

    @property
    def rot(self) -> slice:
        return slice(3, self.pose_dim)

    @property
    def nu_r(self) -> slice:
        return slice(self.pose_dim, self.pose_dim + self.nu_dim)

    @property
    def u_act(self) -> slice:
        start = self.nu_r.stop
        return slice(start, start + self.u_dim)

    @property
    def u_cmd(self) -> slice:
        start = self.u_act.stop
        return slice(start, start + self.u_dim)

    @property
    def v_c(self) -> slice:
        start = self.u_cmd.stop
        return slice(start, start + 3)

    @property
    def depth_ref(self) -> slice:
        start = self.v_c.stop if self.ocean_current else self.u_cmd.stop
        return slice(start, start + 1)

    @property
    def state_dim(self) -> int:
        base = self.v_c.stop if self.ocean_current else self.u_cmd.stop
        return self.depth_ref.stop if self.absolute_depth_context else base


class AUVBaseModel(nn.Module):
    """Shared augmented-state utilities and actuator dynamics."""

    POSE_DIM = 12
    NU_DIM = 6
    V_C_DIM = 3

    def __init__(
        self,
        device=None,
        ocean_current=False,
        T_actuator_init=None,
        u_act_scale=None,
        u_dim=3,
        absolute_depth_context=False,
    ):
        super().__init__()
        self.device = device or torch.device('cpu')
        self.ocean_current = bool(ocean_current)
        self.u_dim = int(u_dim)
        self.absolute_depth_context = bool(absolute_depth_context)
        self.layout = StateLayout(
            pose_dim=self.POSE_DIM,
            nu_dim=self.NU_DIM,
            u_dim=self.u_dim,
            ocean_current=self.ocean_current,
            absolute_depth_context=self.absolute_depth_context,
        )
        self.STATE_DIM = self.layout.state_dim
        self.nfe = 0

        u_act_scale = self._resolve_vector(
            u_act_scale,
            default_last_scale=1e-3,
            fallback_value=1.0,
            name="u_act_scale",
        )
        self.register_buffer(
            'u_act_scale',
            torch.tensor(u_act_scale, dtype=torch.float32, device=self.device),
        )

        T_actuator_init = self._resolve_vector(
            T_actuator_init,
            default_last_scale=1.0,
            fallback_value=0.1,
            name="T_actuator_init",
        )
        raw = self._softplus_inverse(torch.tensor(T_actuator_init, dtype=torch.float32))
        self.T_actuator_raw = nn.Parameter(raw.to(self.device))

    def _resolve_vector(self, values, default_last_scale, fallback_value, name):
        """Resolve a scalar/list parameter to the actuator dimension."""
        if values is None:
            base = [fallback_value] * self.u_dim
            if self.u_dim >= 1:
                base[-1] = default_last_scale
            return base
        if len(values) == self.u_dim:
            return list(values)
        if len(values) == 1:
            return [float(values[0])] * self.u_dim
        raise ValueError(f"{name} must have length 1 or u_dim={self.u_dim}, got {len(values)}.")

    @staticmethod
    def _softplus_inverse(x):
        """Numerically stable inverse: y such that softplus(y) = x."""
        return x + torch.log(-torch.expm1(-x))

    @property
    def T_actuator(self):
        """Actuator time constants, guaranteed positive."""
        return F.softplus(self.T_actuator_raw) + 0.01

    def _parse_state(self, state):
        q = state[:, :self.POSE_DIM]
        nu_r = state[:, self.layout.nu_r]
        u_act = state[:, self.layout.u_act]
        u_cmd = state[:, self.layout.u_cmd]
        v_c_n = state[:, self.layout.v_c] if self.ocean_current else None
        return q, nu_r, u_act, u_cmd, v_c_n

    def _actuator_deriv(self, u_act, u_cmd):
        du_act = (u_cmd - u_act) / self.T_actuator
        du_cmd = torch.zeros_like(u_cmd)
        return du_act, du_cmd

    @staticmethod
    def _body_current(R, v_c_n):
        """Convert inertial-frame current to the body frame via R^T v_c^n."""
        if v_c_n is None:
            return None
        return torch.bmm(R.transpose(1, 2), v_c_n.unsqueeze(-1)).squeeze(-1)

    def _reconstruct_total_velocity(self, nu_r, R, v_c_n):
        """Recover total body-frame velocity from relative velocity and current."""
        if v_c_n is None:
            return nu_r, None

        v_c_body = self._body_current(R, v_c_n)
        nu_total = nu_r.clone()
        nu_total[:, :3] = nu_total[:, :3] + v_c_body
        return nu_total, v_c_body

    def _se3_kinematics(self, R, nu_r, v_c_n=None):
        """Exact SE(3) kinematics in the same convention as AUVHamNODE."""
        nu_total, _ = self._reconstruct_total_velocity(nu_r, R, v_c_n)
        dx = torch.bmm(R, nu_total[:, :3].unsqueeze(-1)).squeeze(-1)
        omega = nu_r[:, 3:]
        dR = torch.cross(R, omega.unsqueeze(1).expand_as(R), dim=2).reshape(-1, 9)
        return dx, dR

    def _shift_linear_velocity(self, state, sign: float):
        """Add/subtract body-frame current from the linear velocity slot."""
        s = state.clone()
        R = s[..., 3:12].reshape(-1, 3, 3)
        v_c_n = s[..., self.layout.v_c].reshape(-1, 3)
        v_c_body = self._body_current(R, v_c_n)
        s_flat = s.reshape(-1, s.shape[-1])
        s_flat[:, 12:15] += sign * v_c_body
        return s_flat.reshape(s.shape)

    def to_ode_state(self, state):
        """Convert data-convention state (nu total) to ODE convention (nu_r)."""
        if not self.ocean_current:
            return state
        return self._shift_linear_velocity(state, -1.0)

    def to_data_state(self, state):
        """Convert ODE-convention state (nu_r) back to data convention (nu total)."""
        if not self.ocean_current:
            return state
        return self._shift_linear_velocity(state, +1.0)

    def _cat_output(self, parts, v_c_n):
        """Concatenate derivatives, appending zero derivatives for carried channels."""
        parts.append(torch.zeros_like(parts[-1][:, :self.u_dim]))
        if v_c_n is not None:
            parts.append(torch.zeros_like(v_c_n))
        if self.absolute_depth_context:
            ref = parts[0][:, :1]
            parts.append(torch.zeros_like(ref))
        return torch.cat(parts, dim=1)

    def _absolute_depth(self, x, state):
        """Recover absolute depth from relative position plus optional context."""
        depth = x[:, 2:3]
        if self.absolute_depth_context:
            depth = depth + state[:, self.layout.depth_ref]
        return depth

    def reset_nfe(self):
        self.nfe = 0


class AUVMassModel(AUVBaseModel):
    """Base class for models with a constant SPD mass matrix."""

    def __init__(self, device=None, M_init=None, **kwargs):
        super().__init__(device=device, **kwargs)
        if M_init is not None:
            self.Minv_L = nn.Parameter(self._init_mass_param(M_init).to(self.device))
        else:
            self.Minv_L = nn.Parameter(
                torch.eye(self.NU_DIM, device=self.device)
                + 0.01 * torch.randn(self.NU_DIM, self.NU_DIM, device=self.device)
            )

    def _init_mass_param(self, M_init):
        """Compute Minv_L from a physical 6x6 mass matrix."""
        mass = torch.as_tensor(M_init, dtype=torch.float32)
        mass_inv = torch.linalg.inv(mass)
        L_target = torch.linalg.cholesky(mass_inv)

        diag_target = torch.diag(L_target)
        sp_input = (diag_target - 1e-4).clamp(min=1e-6)
        diag_raw = self._softplus_inverse(sp_input)

        L_raw = torch.tril(L_target.clone())
        L_raw = L_raw - torch.diag(torch.diag(L_raw)) + torch.diag(diag_raw)
        return L_raw

    @property
    def mass_inv(self):
        """M^{-1} = L L^T, positive-definite by construction."""
        L = torch.tril(self.Minv_L)
        diag = F.softplus(torch.diag(L)) + 1e-4
        L = L - torch.diag(torch.diag(L)) + torch.diag(diag)
        return L @ L.t()

    def _mass_matrix(self):
        M_inv = self.mass_inv
        eye = torch.eye(self.NU_DIM, device=M_inv.device, dtype=M_inv.dtype)
        M = torch.linalg.solve(M_inv, eye)
        return M_inv, M

    def _momentum(self, nu_r, mass):
        return torch.einsum('ij,bj->bi', mass, nu_r)


class AUVHamNODEMergedNC(AUVMassModel):
    """
    Hamiltonian NODE with merged non-conservative forces.

    Keeps explicit quadratic kinetic energy and scalar potential V(q), but
    replaces decomposed D/J/B with a single learned force F_nc(nu_r, u).
    """

    def __init__(
        self,
        device=None,
        hidden_dim=128,
        include_depth_in_potential=False,
        M_init=None,
        T_actuator_init=None,
        u_act_scale=None,
        ocean_current=False,
        u_dim=3,
        absolute_depth_context=False,
        **kwargs,
    ):
        super().__init__(
            device=device,
            ocean_current=ocean_current,
            M_init=M_init,
            T_actuator_init=T_actuator_init,
            u_act_scale=u_act_scale,
            u_dim=u_dim,
            absolute_depth_context=absolute_depth_context,
        )
        self.include_depth_in_potential = include_depth_in_potential

        self.V_linear = nn.Parameter(torch.zeros(1, device=self.device))
        v_in = 4 if include_depth_in_potential else 3
        self.V_net = MLP(v_in, hidden_dim, 1, gain=0.01).to(self.device)

        f_hidden = int(hidden_dim * 1.59)
        f_in = self.NU_DIM + self.u_dim + (3 if ocean_current else 0)
        self.F_net = MLP(f_in, f_hidden, self.NU_DIM, gain=0.1).to(self.device)

    def _potential(self, x, R, depth_ref=None):
        g_body = R.transpose(1, 2)[:, :, 2]
        V_prior = self.V_linear * g_body[:, 2:3]
        inp = (
            torch.cat([g_body, x[:, 2:3] + depth_ref], dim=1)
            if self.include_depth_in_potential else g_body
        )
        return (V_prior + self.V_net(inp)).squeeze(-1)

    def _potential_gradients(self, q, create_graph, depth_ref=None):
        q_in = q.detach().requires_grad_(True) if not q.requires_grad else q
        x = q_in[:, :3]
        R = q_in[:, 3:12].view(-1, 3, 3)
        V = self._potential(x, R, depth_ref=depth_ref)
        dV_dq = torch.autograd.grad(V.sum(), q_in, create_graph=create_graph)[0]
        return dV_dq[:, :3], dV_dq[:, 3:12]

    def forward(self, t, state):
        create_graph = self.training

        with torch.enable_grad():
            self.nfe += 1
            q, nu_r, u_act, u_cmd, v_c_n = self._parse_state(state)
            depth_ref = state[:, self.layout.depth_ref] if self.absolute_depth_context else None

            M_inv, M = self._mass_matrix()
            p_r = self._momentum(nu_r, M)
            dV_dx, dV_dR = self._potential_gradients(q, create_graph=create_graph, depth_ref=depth_ref)

            R = q[:, 3:12].view(-1, 3, 3)
            dx, dR = self._se3_kinematics(R, nu_r, v_c_n)
            _, v_c_body = self._reconstruct_total_velocity(nu_r, R, v_c_n)
            v_r, omega = nu_r[:, :3], nu_r[:, 3:]
            p_rv, p_rw = p_r[:, :3], p_r[:, 3:]

            dp_rv = p_rv.cross(omega, dim=1)
            dp_rw = p_rw.cross(omega, dim=1) + p_rv.cross(v_r, dim=1)

            dp_rv = dp_rv - torch.bmm(R.transpose(1, 2), dV_dx.unsqueeze(-1)).squeeze(-1)
            dV_dR_mat = dV_dR.view(-1, 3, 3)
            dp_rw = dp_rw + torch.cross(R, dV_dR_mat, dim=2).sum(dim=1)

            u_act_scaled = u_act * self.u_act_scale
            parts = [nu_r]
            if v_c_body is not None:
                parts.append(v_c_body)
            parts.append(u_act_scaled)
            f_nc = self.F_net(torch.cat(parts, dim=1))

            dp_r = torch.cat([dp_rv, dp_rw], dim=1) + f_nc
            dnu_r = dp_r @ M_inv
            du_act, du_cmd = self._actuator_deriv(u_act, u_cmd)

            return self._cat_output([dx, dR, dnu_r, du_act], v_c_n)

    @torch.no_grad()
    def energy(self, state):
        q = state[:, :self.POSE_DIM]
        nu_r = state[:, self.layout.nu_r]
        depth_ref = state[:, self.layout.depth_ref] if self.absolute_depth_context else None
        M_inv, M = self._mass_matrix()
        p_r = self._momentum(nu_r, M)
        kinetic = 0.5 * torch.sum(p_r * (p_r @ M_inv), dim=1)
        potential = self._potential(q[:, :3], q[:, 3:12].view(-1, 3, 3), depth_ref=depth_ref)
        return kinetic + potential


class AUVHamNODEQForce(AUVMassModel):
    """
    Structured pH ablation without a scalar potential.

    Keeps exact SE(3), momentum coordinates, co-adjoint structure, and separate
    D/J/B branches. The conservative contribution is replaced by a generic
    configuration-dependent generalized force g_q(q) instead of -dV/dq.
    """

    def __init__(
        self,
        device=None,
        hidden_dim=128,
        include_depth=False,
        M_init=None,
        T_actuator_init=None,
        u_act_scale=None,
        ocean_current=False,
        coupled_damping=True,
        learn_lift=True,
        actuation_condition_on_velocity=True,
        actuation_current_feature="current_body",
        dj_current_feature="none",
        u_dim=3,
        absolute_depth_context=False,
        **kwargs,
    ):
        super().__init__(
            device=device,
            ocean_current=ocean_current,
            M_init=M_init,
            T_actuator_init=T_actuator_init,
            u_act_scale=u_act_scale,
            u_dim=u_dim,
            absolute_depth_context=absolute_depth_context,
        )
        valid_actuation_features = {"none", "current_body", "total_velocity"}
        valid_dj_features = {"none", "current_body", "total_velocity"}
        if actuation_current_feature not in valid_actuation_features:
            raise ValueError(
                "actuation_current_feature must be one of "
                f"{sorted(valid_actuation_features)}, got {actuation_current_feature!r}."
            )
        if dj_current_feature not in valid_dj_features:
            raise ValueError(
                "dj_current_feature must be one of "
                f"{sorted(valid_dj_features)}, got {dj_current_feature!r}."
            )

        self.include_depth = include_depth
        self.coupled_damping = coupled_damping
        self.learn_lift = learn_lift
        self.actuation_condition_on_velocity = actuation_condition_on_velocity
        self.actuation_current_feature = actuation_current_feature if ocean_current else "none"
        self.dj_current_feature = dj_current_feature if ocean_current else "none"

        q_in = 4 if include_depth else 3
        self.G_net = MLP(q_in, hidden_dim, self.NU_DIM, gain=0.05).to(self.device)

        dj_in = self.NU_DIM + (3 if self.dj_current_feature != "none" else 0)
        d_out = 21 if coupled_damping else 6
        self.D_net = MLP(dj_in, hidden_dim, d_out, gain=0.01).to(self.device)
        self.J_net = (
            MLP(dj_in, hidden_dim, 15, gain=0.01).to(self.device)
            if learn_lift else None
        )

        b_in = self.u_dim
        if self.actuation_condition_on_velocity:
            b_in += self.NU_DIM
        if self.actuation_current_feature != "none":
            b_in += 3
        self.B_net = MLP(b_in, hidden_dim // 2, self.NU_DIM, gain=0.1).to(self.device)

    def _q_force(self, x, R, depth_ref=None):
        """Configuration-dependent generalized force in body momentum coordinates."""
        g_body = R.transpose(1, 2)[:, :, 2]
        inp = (
            torch.cat([g_body, x[:, 2:3] + depth_ref], dim=1)
            if self.include_depth else g_body
        )
        return self.G_net(inp)

    def damping(self, nu_r, extra=None):
        bs, dev = nu_r.shape[0], nu_r.device
        net_in = nu_r if extra is None else torch.cat([nu_r, extra], dim=1)
        if self.coupled_damping:
            raw = self.D_net(net_in)
            L = torch.zeros(bs, self.NU_DIM, self.NU_DIM, device=dev, dtype=nu_r.dtype)
            diag_idx = range(self.NU_DIM)
            L[:, diag_idx, diag_idx] = F.softplus(raw[:, :self.NU_DIM]) + 0.01
            idx = torch.tril_indices(self.NU_DIM, self.NU_DIM, offset=-1, device=dev)
            L[:, idx[0], idx[1]] = raw[:, self.NU_DIM:]
            return torch.bmm(L, L.transpose(1, 2))
        d = F.softplus(self.D_net(net_in)) + 0.01
        return torch.diag_embed(d)

    def lift(self, nu_r, extra=None):
        if self.J_net is None:
            bs = nu_r.shape[0]
            return torch.zeros(bs, self.NU_DIM, self.NU_DIM, device=nu_r.device, dtype=nu_r.dtype)

        bs = nu_r.shape[0]
        net_in = nu_r if extra is None else torch.cat([nu_r, extra], dim=1)
        raw = self.J_net(net_in)
        A = torch.zeros(bs, self.NU_DIM, self.NU_DIM, device=nu_r.device, dtype=nu_r.dtype)
        idx = torch.triu_indices(self.NU_DIM, self.NU_DIM, offset=1, device=nu_r.device)
        A[:, idx[0], idx[1]] = raw
        return A - A.transpose(1, 2)

    def forward(self, t, state):
        with torch.enable_grad():
            self.nfe += 1
            q, nu_r, u_act, u_cmd, v_c_n = self._parse_state(state)
            depth_ref = state[:, self.layout.depth_ref] if self.absolute_depth_context else None

            M_inv, M = self._mass_matrix()
            p_r = self._momentum(nu_r, M)

            x = q[:, :3]
            R = q[:, 3:12].view(-1, 3, 3)
            nu_total, v_c_body = self._reconstruct_total_velocity(nu_r, R, v_c_n)
            dx, dR = self._se3_kinematics(R, nu_r, v_c_n)

            v_r, omega = nu_r[:, :3], nu_r[:, 3:]
            p_rv, p_rw = p_r[:, :3], p_r[:, 3:]
            dp_rv = p_rv.cross(omega, dim=1)
            dp_rw = p_rw.cross(omega, dim=1) + p_rv.cross(v_r, dim=1)

            q_force = self._q_force(x, R, depth_ref=depth_ref)

            u_act_scaled = u_act * self.u_act_scale
            tau_in = [u_act_scaled]
            if self.actuation_condition_on_velocity:
                tau_in.insert(0, nu_r)
            if self.actuation_current_feature == "current_body":
                tau_in.append(v_c_body)
            elif self.actuation_current_feature == "total_velocity":
                tau_in.append(nu_total[:, :3])
            tau = self.B_net(torch.cat(tau_in, dim=1))

            if self.dj_current_feature == "current_body":
                d_input = v_c_body
            elif self.dj_current_feature == "total_velocity":
                d_input = nu_total[:, :3]
            else:
                d_input = None
            f_nc = (
                -torch.bmm(self.damping(nu_r, d_input), nu_r.unsqueeze(-1)).squeeze(-1)
                + torch.bmm(self.lift(nu_r, d_input), nu_r.unsqueeze(-1)).squeeze(-1)
                + tau
            )

            dp_r = torch.cat([dp_rv, dp_rw], dim=1) + q_force + f_nc
            dnu_r = dp_r @ M_inv
            du_act, du_cmd = self._actuator_deriv(u_act, u_cmd)

            return self._cat_output([dx, dR, dnu_r, du_act], v_c_n)

    @torch.no_grad()
    def energy(self, state):
        nu_r = state[:, self.layout.nu_r]
        M_inv, M = self._mass_matrix()
        p_r = self._momentum(nu_r, M)
        return 0.5 * torch.sum(p_r * (p_r @ M_inv), dim=1)


class AUVMomentumSE3NODE(AUVMassModel):
    """SE(3) NODE in momentum coordinates with a constant SPD mass matrix."""

    def __init__(
        self,
        device=None,
        hidden_dim=128,
        include_depth=False,
        M_init=None,
        T_actuator_init=None,
        u_act_scale=None,
        ocean_current=False,
        u_dim=3,
        absolute_depth_context=False,
        **kwargs,
    ):
        super().__init__(
            device=device,
            ocean_current=ocean_current,
            M_init=M_init,
            T_actuator_init=T_actuator_init,
            u_act_scale=u_act_scale,
            u_dim=u_dim,
            absolute_depth_context=absolute_depth_context,
        )
        self.include_depth = include_depth

        f_hidden = int(hidden_dim * 1.90)
        f_in = self.NU_DIM + 3 + self.u_dim + (1 if include_depth else 0)
        if ocean_current:
            f_in += 3
        self.F_net = MLP(f_in, f_hidden, self.NU_DIM, gain=0.1).to(self.device)

    def forward(self, t, state):
        with torch.enable_grad():
            self.nfe += 1
            q, nu_r, u_act, u_cmd, v_c_n = self._parse_state(state)

            M_inv, M = self._mass_matrix()
            p_r = self._momentum(nu_r, M)

            R = q[:, 3:12].view(-1, 3, 3)
            dx, dR = self._se3_kinematics(R, nu_r, v_c_n)
            _, v_c_body = self._reconstruct_total_velocity(nu_r, R, v_c_n)

            g_body = R.transpose(1, 2)[:, :, 2]
            u_act_scaled = u_act * self.u_act_scale

            parts = [p_r, g_body]
            if self.include_depth:
                parts.append(self._absolute_depth(q[:, :3], state))
            if v_c_body is not None:
                parts.append(v_c_body)
            parts.append(u_act_scaled)
            dp_r = self.F_net(torch.cat(parts, dim=1))
            dnu_r = dp_r @ M_inv

            du_act, du_cmd = self._actuator_deriv(u_act, u_cmd)
            return self._cat_output([dx, dR, dnu_r, du_act], v_c_n)

    @torch.no_grad()
    def energy(self, state):
        nu_r = state[:, self.layout.nu_r]
        M_inv, M = self._mass_matrix()
        p_r = self._momentum(nu_r, M)
        return 0.5 * torch.sum(p_r * (p_r @ M_inv), dim=1)


class AUVUnstructuredHamNODE(AUVMassModel):
    """SE(3) Hamiltonian NODE with a single learned energy function."""

    def __init__(
        self,
        device=None,
        hidden_dim=128,
        M_init=None,
        T_actuator_init=None,
        u_act_scale=None,
        ocean_current=False,
        u_dim=3,
        absolute_depth_context=False,
        **kwargs,
    ):
        super().__init__(
            device=device,
            ocean_current=ocean_current,
            M_init=M_init,
            T_actuator_init=T_actuator_init,
            u_act_scale=u_act_scale,
            u_dim=u_dim,
            absolute_depth_context=absolute_depth_context,
        )

        h_hidden = int(hidden_dim * 1.55)
        self.H_net = MLP(18, h_hidden, 1, gain=0.01).to(self.device)

        f_in = self.NU_DIM + self.u_dim + (3 if ocean_current else 0)
        self.F_net = MLP(f_in, hidden_dim, self.NU_DIM, gain=0.1).to(self.device)

    def forward(self, t, state):
        create_graph = self.training

        with torch.enable_grad():
            self.nfe += 1
            q, nu_r, u_act, u_cmd, v_c_n = self._parse_state(state)

            M_inv, M = self._mass_matrix()
            p_r = self._momentum(nu_r, M)

            qp = torch.cat([q, p_r], dim=1)
            if not qp.requires_grad:
                qp = qp.detach().requires_grad_(True)

            H = self.H_net(qp).squeeze(-1)
            dH = torch.autograd.grad(H.sum(), qp, create_graph=create_graph)[0]

            dH_dx = dH[:, :3]
            dH_dR = dH[:, 3:12]
            dH_dpv = dH[:, 12:15]
            dH_dpw = dH[:, 15:18]

            R = q[:, 3:12].view(-1, 3, 3)
            p_rv, p_rw = p_r[:, :3], p_r[:, 3:]
            v_c_body = self._body_current(R, v_c_n)

            linear_velocity = dH_dpv
            if v_c_body is not None:
                linear_velocity = linear_velocity + v_c_body
            dx = torch.bmm(R, linear_velocity.unsqueeze(-1)).squeeze(-1)
            dR = torch.cross(R, dH_dpw.unsqueeze(1).expand_as(R), dim=2).reshape(-1, 9)

            dp_rv = p_rv.cross(dH_dpw, dim=1) - torch.bmm(
                R.transpose(1, 2),
                dH_dx.unsqueeze(-1),
            ).squeeze(-1)

            dH_dR_mat = dH_dR.view(-1, 3, 3)
            dp_rw = (
                p_rw.cross(dH_dpw, dim=1)
                + p_rv.cross(dH_dpv, dim=1)
                + torch.cross(R, dH_dR_mat, dim=2).sum(dim=1)
            )

            u_act_scaled = u_act * self.u_act_scale
            parts = [nu_r]
            if v_c_body is not None:
                parts.append(v_c_body)
            parts.append(u_act_scaled)
            f_nc = self.F_net(torch.cat(parts, dim=1))

            dp_r = torch.cat([dp_rv, dp_rw], dim=1) + f_nc
            dnu_r = dp_r @ M_inv
            du_act, du_cmd = self._actuator_deriv(u_act, u_cmd)

            return self._cat_output([dx, dR, dnu_r, du_act], v_c_n)

    @torch.no_grad()
    def energy(self, state):
        q = state[:, :self.POSE_DIM]
        nu_r = state[:, self.layout.nu_r]
        _, M = self._mass_matrix()
        p_r = self._momentum(nu_r, M)
        qp = torch.cat([q, p_r], dim=1)
        return self.H_net(qp).squeeze(-1)


class AUVUnstructuredSE3NODE(AUVBaseModel):
    """SE(3) NODE with exact kinematics and black-box acceleration."""

    def __init__(
        self,
        device=None,
        hidden_dim=240,
        include_depth=False,
        T_actuator_init=None,
        u_act_scale=None,
        ocean_current=False,
        u_dim=3,
        absolute_depth_context=False,
        **kwargs,
    ):
        super().__init__(
            device=device,
            ocean_current=ocean_current,
            T_actuator_init=T_actuator_init,
            u_act_scale=u_act_scale,
            u_dim=u_dim,
            absolute_depth_context=absolute_depth_context,
        )
        self.include_depth = include_depth

        in_dim = self.NU_DIM + 3 + self.u_dim + (1 if include_depth else 0)
        if ocean_current:
            in_dim += 3
        self.f_net = MLP(in_dim, hidden_dim, self.NU_DIM, gain=0.1).to(self.device)

    def forward(self, t, state):
        with torch.enable_grad():
            self.nfe += 1
            q, nu_r, u_act, u_cmd, v_c_n = self._parse_state(state)

            R = q[:, 3:12].view(-1, 3, 3)
            dx, dR = self._se3_kinematics(R, nu_r, v_c_n)
            _, v_c_body = self._reconstruct_total_velocity(nu_r, R, v_c_n)

            g_body = R.transpose(1, 2)[:, :, 2]
            u_act_scaled = u_act * self.u_act_scale

            parts = [nu_r, g_body]
            if v_c_body is not None:
                parts.append(v_c_body)
            parts.append(u_act_scaled)
            if self.include_depth:
                parts.append(self._absolute_depth(q[:, :3], state))
            dnu_r = self.f_net(torch.cat(parts, dim=1))

            du_act, du_cmd = self._actuator_deriv(u_act, u_cmd)
            return self._cat_output([dx, dR, dnu_r, du_act], v_c_n)


class AUVBlackBoxNODE(AUVBaseModel):
    """Fully unstructured NODE, including learned kinematics."""

    def __init__(
        self,
        device=None,
        hidden_dim=228,
        T_actuator_init=None,
        u_act_scale=None,
        ocean_current=False,
        u_dim=3,
        absolute_depth_context=False,
        **kwargs,
    ):
        super().__init__(
            device=device,
            ocean_current=ocean_current,
            T_actuator_init=T_actuator_init,
            u_act_scale=u_act_scale,
            u_dim=u_dim,
            absolute_depth_context=absolute_depth_context,
        )

        in_dim = self.POSE_DIM + self.NU_DIM + self.u_dim + (3 if ocean_current else 0)
        out_dim = self.POSE_DIM + self.NU_DIM
        self.f_net = MLP(in_dim, hidden_dim, out_dim, gain=0.1).to(self.device)

    def forward(self, t, state):
        with torch.enable_grad():
            self.nfe += 1
            q, nu_r, u_act, u_cmd, v_c_n = self._parse_state(state)

            R = q[:, 3:12].view(-1, 3, 3)
            v_c_body = self._body_current(R, v_c_n)
            u_act_scaled = u_act * self.u_act_scale
            parts = [q, nu_r, u_act_scaled]
            if v_c_body is not None:
                parts.append(v_c_body)
            dq_dnu = self.f_net(torch.cat(parts, dim=1))

            du_act, du_cmd = self._actuator_deriv(u_act, u_cmd)
            return self._cat_output([dq_dnu, du_act], v_c_n)


BASELINE_MODELS = {
    'merged_nc': AUVHamNODEMergedNC,
    'hamnode_no_potential': AUVHamNODEQForce,
    'hamnode_qforce': AUVHamNODEQForce,
    'no_potential': AUVHamNODEQForce,
    'momentum_se3': AUVMomentumSE3NODE,
    'unstructured_ham': AUVUnstructuredHamNODE,
    'unstructured_se3': AUVUnstructuredSE3NODE,
    'blackbox': AUVBlackBoxNODE,
}
