/**
 * Auth & Security TypeScript Tipleri
 */

export type UserRole = 'admin' | 'researcher' | 'viewer';

export interface User {
  user_id: string;
  username: string;
  email: string;
  role: UserRole;
  full_name?: string | null;
  created_at: string;
  last_login?: string | null;
}

export interface UserPermissions {
  can_train: boolean;
  can_delete_models: boolean;
  can_manage_users: boolean;
  can_create_api_keys: boolean;
  can_run_benchmarks: boolean;
  can_view_metrics: boolean;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface RefreshTokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
}

export interface CurrentUserResponse {
  user: User;
  permissions: UserPermissions;
}

export interface APIKeyItem {
  key_id: string;
  user_id: string;
  name: string;
  key_prefix: string;
  role: string;
  is_active: boolean;
  created_at: string;
  expires_at?: string | null;
  last_used_at?: string | null;
  raw_key?: string;
}

export interface CreateAPIKeyRequest {
  name: string;
  role?: string;
  expires_in_days?: number;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  full_name?: string;
  role?: string;
}
